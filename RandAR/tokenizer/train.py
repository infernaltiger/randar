import torch
print(torch.cuda.is_available())  # должно быть True
print(torch.cuda.get_device_name(0))  # должно быть "NVIDIA GeForce RTX 3080 Ti"

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid

from model import Model

def main(args):
    training_data = datasets.CIFAR10(root="data", train=True, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
                                  ]))

    validation_data = datasets.CIFAR10(root="data", train=False, download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
                                    ]))

    data_variance = np.var(training_data.data / 255.0)

    training_loader = DataLoader(training_data, 
                             batch_size=args.batch_size, 
                             shuffle=True,
                             pin_memory=True)
    
    validation_loader = DataLoader(validation_data,
                               batch_size=32,
                               shuffle=True,
                               pin_memory=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = Model(args.num_hiddens, args.num_residual_layers, args.num_residual_hiddens,
              args.num_embeddings, args.embedding_dim, 
              args.commitment_cost, args.decay).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=False)

    model.train()
    train_res_recon_error = []
    train_res_perplexity = []

    for i in range(args.num_training_updates):
        (data, _) = next(iter(training_loader))
        data = data.to(device)
        optimizer.zero_grad()

        vq_loss, data_recon, perplexity = model(data)
        recon_error = F.mse_loss(data_recon, data) / data_variance
        loss = recon_error + vq_loss
        loss.backward()

        optimizer.step()
        
        train_res_recon_error.append(recon_error.item())
        train_res_perplexity.append(perplexity.item())

        if (i+1) % 100 == 0:
            print('%d iterations' % (i+1))
            print('recon_error: %.3f' % np.mean(train_res_recon_error[-100:]))
            print('perplexity: %.3f' % np.mean(train_res_perplexity[-100:]))
            print()

    torch.save(model.state_dict(), "vqvae_cifar10.pth")

    train_res_recon_error_smooth = savgol_filter(train_res_recon_error, 201, 7)
    train_res_perplexity_smooth = savgol_filter(train_res_perplexity, 201, 7)

    f = plt.figure(figsize=(16,8))
    ax = f.add_subplot(1,2,1)
    ax.plot(train_res_recon_error_smooth)
    ax.set_yscale('log')
    ax.set_title('Smoothed NMSE.')
    ax.set_xlabel('iteration')

    ax = f.add_subplot(1,2,2)
    ax.plot(train_res_perplexity_smooth)
    ax.set_title('Smoothed Average codebook usage (perplexity).')
    ax.set_xlabel('iteration')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-training-updates", type=int, default=15000)
    parser.add_argument("--num-hiddens", type=int, default=128)
    parser.add_argument("--num-residual-hiddens", type=int, default=32)
    parser.add_argument("--num-residual-layers", type=int, default=2)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--num-embeddings", type=int, default=512)
    parser.add_argument("--commitment-cost", type=float, default=0.25)
    parser.add_argument("--decay", type=float, default=0.99)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()
    main(args)