import torch
import torchvision
import cv2
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp


import torch
import torchvision
import cv2
import numpy as np

def make_grid(imgs, scale=1.0, row_first=True):
    """
    Args:
        imgs: Tensor [B, C, H, W] (float, range 0-1) OR Numpy [B, H, W, C]
    Returns:
        img_grid: np.ndarray, [H', W', 3], dtype=uint8, range 0-255
    """
    # 1. Конвертация в Tensor [B, C, H, W]
    if isinstance(imgs, np.ndarray):
        if imgs.ndim == 4 and imgs.shape[-1] in (1, 3):
            # Если numpy [B, H, W, C], переводим в [B, C, H, W]
            imgs = torch.from_numpy(imgs).permute(0, 3, 1, 2).float()
        else:
            # Если уже что-то другое, пробуем привести
            imgs = torch.from_numpy(imgs).float()
    elif isinstance(imgs, torch.Tensor):
        imgs = imgs.detach().cpu().float()
    else:
        raise ValueError(f"Unsupported input type: {type(imgs)}")

    # Убедимся, что значения в [0, 1]
    imgs = torch.clamp(imgs, 0, 1)

    B, C, H, W = imgs.shape
    
    # 2. Вычисляем сетку
    num_row = int(np.sqrt(B / 2))
    if num_row < 1:
        num_row = 1
    num_col = int(np.ceil(B / num_row))
    nrow = num_col if row_first else num_row

    # 3. Создаем сетку через torchvision (ожидает [B, C, H, W])
    grid = torchvision.utils.make_grid(imgs, nrow=nrow, padding=0, normalize=False)
    # grid: [C, H_grid, W_grid], float, range 0-1

    # 4. Конвертация в numpy [H, W, C]
    grid_np = grid.permute(1, 2, 0).cpu().numpy()  # [H_grid, W_grid, C]

    # 5. Масштабирование через OpenCV
    if scale != 1.0:
        h, w = grid_np.shape[:2]
        new_size = (int(w * scale), int(h * scale))
        # cv2.resize работает с float, но лучше ресайзить до конвертации в uint8
        grid_np = cv2.resize(grid_np, new_size, interpolation=cv2.INTER_AREA)

    # 6. Конвертация в uint8 [0, 255] для WandB
    grid_np = (grid_np * 255.0).clip(0, 255).astype(np.uint8)

    return grid_np