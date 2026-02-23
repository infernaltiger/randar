import os
import sys
import time
import shutil
import argparse
from copy import deepcopy
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from omegaconf import OmegaConf
from tqdm import tqdm
import wandb

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Make repo imports work when called from repo root.
sys.path.append("./RandAR")

from RandAR_in.util import instantiate_from_config
from RandAR_in.dataset.builder import build_dataset
from RandAR_in.utils.visualization import make_grid
from RandAR_in.utils.logger import create_logger
from RandAR_in.utils.lr_scheduler import get_scheduler

from tokenizer.model import Model as VQModel


def cycle(dl: DataLoader):
    """Loop over the dataloader indefinitely (same as original)."""
    while True:
        for data in dl:
            yield data


def set_seed(seed: int):
    """Single-process seed helper."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def main(args):
    assert torch.cuda.is_available(), "Requires at least one CUDA GPU."
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # -------------------------
    # Load config
    # -------------------------
    config = OmegaConf.load(args.config)

    # Allow overriding via CLI (keeps your original args semantics)
    if args.max_iters is not None:
        config.max_iters = args.max_iters
    if args.global_seed is not None:
        config.global_seed = args.global_seed

    set_seed(config.global_seed)

    # -------------------------
    # Experiment directory
    # -------------------------
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_name = timestamp + f"_bs_{config.global_batch_size}_lr_{config.optimizer.lr}"
    experiment_dir = os.path.join(args.results_dir, exp_name)
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")

    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    logger = create_logger(experiment_dir)
    logger.info(f"Experiment directory: {experiment_dir}")
    logger.info(f"Checkpoint directory: {checkpoint_dir}")
    logger.info(f"Device: {device}")
    logger.info(f"Seed: {config.global_seed}")

    # -------------------------
    # Dataset / Dataloader
    # -------------------------
    # IMPORTANT:
    # - For latent dataset, you should ensure the dataset returns LONG token ids.
    # - If your INatLatentDataset still uses transforms.ToTensor(), fix it to torch.long.
    dataset = build_dataset(is_train=True, args=args, transform=transforms.ToTensor())

    # Single GPU => per_gpu_batch_size is just global_batch_size / grad_accum
    grad_accum = int(config.accelerator.gradient_accumulation_steps)
    assert grad_accum >= 1
    per_gpu_batch_size = int(config.global_batch_size // grad_accum)

    data_loader = DataLoader(
        dataset,
        batch_size=per_gpu_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        # Windows-friendly defaults:
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    data_loader = cycle(data_loader)

    logger.info(f"Dataset contains {len(dataset)} samples.")
    logger.info(f"Per-step batch size (on cuda:0): {per_gpu_batch_size}")
    logger.info(f"Grad accumulation steps: {grad_accum}")
    logger.info(f"Effective global batch size: {per_gpu_batch_size * grad_accum}")

    # -------------------------
    # Model
    # -------------------------
    model = instantiate_from_config(config.ar_model).to(device)
    logger.info(f"GPT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    model.train()

    # -------------------------
    # Tokenizer (your CIFAR VQ-VAE)
    # -------------------------
    # FIX: original script used ckpt_path (undefined). Use args.vq_ckpt instead.  :contentReference[oaicite:2]{index=2}
    tokenizer = VQModel(128, 2, 32, 512, 64, 0.25, 0.99).to(device)
    tokenizer.load_state_dict(torch.load(args.vq_ckpt, map_location=device))
    tokenizer.eval()
    for p in tokenizer.parameters():
        p.requires_grad_(False)

    # -------------------------
    # Optimizer / LR scheduler
    # -------------------------
    optimizer = model.configure_optimizer(**config.optimizer)

    # In original code, num_training_steps was multiplied by num_processes (DDP).
    # Here num_processes = 1.
    lr_scheduler = get_scheduler(
        name=config.lr_scheduler.type,
        optimizer=optimizer,
        num_warmup_steps=config.lr_scheduler.warm_up_iters * grad_accum,
        num_training_steps=config.max_iters * grad_accum,
        min_lr_ratio=config.lr_scheduler.min_lr_ratio,
        num_cycles=config.lr_scheduler.num_cycles,
    )

    # -------------------------
    # W&B
    # -------------------------
    os.environ["WANDB__SERVICE_WAIT"] = "600"
    if args.wandb_offline:
        os.environ["WANDB_MODE"] = "offline"

    wandb_run = wandb.init(
        project="RandAR-Release",
        entity=args.wandb_entity,
        config=dict(config),
        name=exp_name,
        dir=experiment_dir,
    )

    # -------------------------
    # Resume training (simple, single-process)
    # -------------------------
    # Your previous code resumed via accelerator.save_state/load_state; we replace with
    # manual load/save of model+optimizer+lr_scheduler+step.
    #
    # It will resume from the latest "iters_XXXXXXXX" folder if found.
    train_steps = 0
    saved_ckpt_dirs = [d for d in os.listdir(checkpoint_dir) if d.startswith("iters_")]
    if len(saved_ckpt_dirs) > 0:
        saved_ckpt_dirs = sorted(saved_ckpt_dirs)
        last_dir = saved_ckpt_dirs[-1]
        ckpt_dir = os.path.join(checkpoint_dir, last_dir)

        ckpt_file = os.path.join(ckpt_dir, "train_state.pt")
        if os.path.exists(ckpt_file):
            logger.info(f"Resuming from {ckpt_file}")
            state = torch.load(ckpt_file, map_location="cpu")
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            lr_scheduler.load_state_dict(state["lr_scheduler"])
            train_steps = int(state["train_steps"])
        else:
            logger.info(f"Found {ckpt_dir} but no train_state.pt; starting from scratch.")

    # -------------------------
    # Training loop
    # -------------------------
    total_iters = int(config.max_iters)
    logger.info(f"Starting training from iteration {train_steps} to {total_iters}")

    log_every = int(args.log_every)
    ckpt_every = int(args.ckpt_every)
    visualize_every = int(args.visualize_every)

    running_loss = 0.0
    running_grad_norm = 0.0
    start_time = time.time()

    scaler = None
    use_amp = (args.mixed_precision in ["fp16", "bf16"])
    amp_dtype = torch.float16 if args.mixed_precision == "fp16" else (torch.bfloat16 if args.mixed_precision == "bf16" else None)
    if args.mixed_precision == "fp16":
        scaler = torch.cuda.amp.GradScaler()

    # local helper to compute "grad norm" similarly to your original loop
    def compute_grad_norm_l2(m: torch.nn.Module) -> float:
        total = 0.0
        for p in m.parameters():
            if p.grad is None:
                continue
            total += p.grad.data.norm(2).item()
        return total

    while train_steps < total_iters:
        model.train()

        # Gradient accumulation loop
        optimizer.zero_grad(set_to_none=True)

        loss_this_step = 0.0
        grad_norm_this_step = 0.0

        for micro in range(grad_accum):
            x, y, _inat_index = next(data_loader)

            # NOTE: your latent dataset should already return token IDs.
            # x typically has shape (B, 1, T). Your code flattens it.
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            image_tokens = x.reshape(x.shape[0], -1)
            cond = y.reshape(-1)

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                logits, loss, token_order = model(image_tokens, cond, targets=image_tokens)

            if scaler is not None:
                scaler.scale(loss / grad_accum).backward()
            else:
                (loss / grad_accum).backward()

            loss_this_step += float(loss.detach().item())

        # gradient clipping + step
        if config.optimizer.max_grad_norm != 0.0:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimizer.max_grad_norm)

        grad_norm = compute_grad_norm_l2(model)
        grad_norm_this_step = grad_norm

        # mimic your "skip grad norm" logic
        if grad_norm < config.optimizer.skip_grad_norm or train_steps < config.optimizer.skip_grad_iter:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

        lr_scheduler.step()

        # bookkeeping
        running_loss += (loss_this_step / grad_accum)
        running_grad_norm += grad_norm_this_step

        train_steps += 1

        # -------------------------
        # Logging
        # -------------------------
        if train_steps % log_every == 0:
            avg_loss = running_loss / log_every
            avg_grad = running_grad_norm / log_every

            end_time = time.time()
            avg_time = (end_time - start_time) / log_every
            start_time = time.time()

            lr = lr_scheduler.get_last_lr()[0]
            logger.info(
                f"Step {train_steps:08d} | Loss {avg_loss:.4f} | Time {avg_time:.4f}s | "
                f"Grad Norm {avg_grad:.4f} | LR {lr:.6f}"
            )

            wandb.log(
                {
                    "loss": avg_loss,
                    "benchmark/time": avg_time,
                    "grad_norm": avg_grad,
                    "lr": lr,
                },
                step=train_steps,
            )

            running_loss = 0.0
            running_grad_norm = 0.0

        # -------------------------
        # Visualization (teacher forcing / gt recon / generation)
        # -------------------------
        if visualize_every > 0 and (train_steps % visualize_every == 0):
            model.eval()
            with torch.no_grad():
                visualize_num = int(args.visualize_num)

                visualize_logits = logits[:visualize_num]
                visualize_cond = cond[:visualize_num]
                visualize_token_order = token_order[:visualize_num]
                visualize_gt_indices = image_tokens[:visualize_num]

                orig_token_order = torch.argsort(visualize_token_order)
                img_token_num = visualize_logits.shape[1]

                # teacher forcing reconstruction
                pred_recon_indices = torch.zeros(
                    visualize_num, img_token_num, device=device, dtype=torch.long
                )
                for i in range(img_token_num):
                    pred_recon_indices[:, i : i + 1] = torch.argmax(
                        visualize_logits[:, i : i + 1], dim=-1
                    )

                pred_recon_indices = torch.gather(
                    pred_recon_indices.unsqueeze(-1),
                    dim=1,
                    index=orig_token_order.unsqueeze(-1),
                ).squeeze(-1)

                pred_recon_imgs = tokenizer.decode_codes_to_img(pred_recon_indices, args.image_size)

                # VQ reconstruction from ground truth codes
                gt_recon_imgs = tokenizer.decode_codes_to_img(visualize_gt_indices, args.image_size)

                # generation
                # FIX: in single GPU, do NOT use model.module.generate  :contentReference[oaicite:3]{index=3}
                gen_indices = model.generate(
                    cond=visualize_cond,
                    token_order=None,
                    cfg_scales=[4.0, 4.0],
                    num_inference_steps=-1,
                    temperature=1.0,
                    top_k=0,
                    top_p=1.0,
                )
                model.remove_caches()
                gen_imgs = tokenizer.decode_codes_to_img(gen_indices, args.image_size)

                pred_recon_grid = make_grid(pred_recon_imgs)
                gt_recon_grid = make_grid(gt_recon_imgs)
                gen_grid = make_grid(gen_imgs)

                wandb.log(
                    {
                        "pred_recon": wandb.Image(pred_recon_grid),
                        "gt_recon": wandb.Image(gt_recon_grid),
                        "gen": wandb.Image(gen_grid),
                    },
                    step=train_steps,
                )
            model.train()

        # -------------------------
        # Checkpointing
        # -------------------------
        if ckpt_every > 0 and (train_steps % ckpt_every == 0):
            ckpt_path = os.path.join(checkpoint_dir, f"iters_{train_steps:08d}")
            os.makedirs(ckpt_path, exist_ok=True)

            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "train_steps": train_steps,
                "config": dict(config),
            }
            torch.save(state, os.path.join(ckpt_path, "train_state.pt"))
            logger.info(f"Saved Iter {train_steps} checkpoint to {ckpt_path}")

            # remove older checkpoints (same policy as your code)
            for ckpt_dir in os.listdir(checkpoint_dir):
                if ckpt_dir.startswith("iters_") and ckpt_dir != f"iters_{train_steps:08d}":
                    save_iter = int(ckpt_dir.split("_")[-1])
                    if save_iter < train_steps - args.keep_last_k * ckpt_every:
                        if save_iter not in [50000, 100000, 200000, 300000]:
                            shutil.rmtree(os.path.join(checkpoint_dir, ckpt_dir), ignore_errors=True)

            # optional disk copy
            if args.disk_location:
                disk_location = os.path.join(args.disk_location, exp_name)
                try:
                    if os.path.exists(disk_location):
                        shutil.rmtree(disk_location)
                    shutil.copytree(checkpoint_dir, disk_location)
                    logger.info(f"Copied checkpoint to {disk_location}")
                except Exception as e:
                    logger.error(f"Error copying checkpoint to {disk_location}: {e}")

    # final save
    final_ckpt_dir = os.path.join(checkpoint_dir, f"iters_{train_steps:08d}_final")
    os.makedirs(final_ckpt_dir, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "train_steps": train_steps,
        "config": dict(config),
    }
    torch.save(state, os.path.join(final_ckpt_dir, "train_state.pt"))
    logger.info(f"Saved Final Iter {train_steps} checkpoint to {final_ckpt_dir}")

    wandb_run.finish()
    logger.info("Training Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="RandAR/configs/randar/randar_cifar10.yaml")
    parser.add_argument("--results-dir", type=str, default="results")

    # CIFAR-friendly
    parser.add_argument("--image-size", type=int, choices=[32, 128, 256, 384, 448, 512], default=32)
    parser.add_argument("--num-classes", type=int, default=10)

    # Training
    parser.add_argument("--max-iters", type=int, default=None)
    parser.add_argument("--global-seed", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    parser.add_argument("--keep-last-k", type=int, default=1)
    parser.add_argument("--mixed-precision", type=str, default="bf16", choices=["none", "fp16", "bf16"])

    # Tokenizer ckpt
    parser.add_argument("--vq-ckpt", type=str, default="RandAR/tokenizer/vqvae_cifar10.pth")

    # Data
    parser.add_argument("--dataset", type=str, default="latent")
    parser.add_argument("--data-path", type=str, default="data/latents_cifar_10/cifar10-cifar10-cifar10-32_codes")

    # Visualization
    parser.add_argument("--visualize-every", type=int, default=2000)
    parser.add_argument("--visualize-num", type=int, default=32)

    # W&B
    parser.add_argument("--wandb-entity", type=str, default="RandAR")
    parser.add_argument("--wandb-offline", action="store_true")
    parser.add_argument("--disk-location", type=str, default="")

    args = parser.parse_args()

    if args.wandb_offline:
        os.environ["WANDB_MODE"] = "offline"

    main(args)