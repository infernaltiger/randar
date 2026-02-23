import os
from torch.utils import data
from torchvision import datasets, transforms
import torch
import numpy as np

class CIFAR10WithIndex(data.Dataset):
    """
    Wraps torchvision.datasets.CIFAR10 to return (image, label, index),
    matching RandAR training loop expectations.
    """
    def __init__(self, root, train=True, transform=None, download=False):
        self.ds = datasets.CIFAR10(
            root=root,
            train=train,
            transform=transform,
            download=download,
        )
        self.nb_classes = 10  # RandAR code sometimes reads this field

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        img, label = self.ds[index]
        return img, label, index


class CifarLatentDataset(data.Dataset):
    def __init__(self, root_dir, transform=transforms.ToTensor()):
        categories_set = set()
        self.categories = sorted([int(i) for i in list(os.listdir(root_dir))])
        self.samples = []

        for tgt_class in self.categories:
            tgt_dir = os.path.join(root_dir, str(tgt_class))
            for root, _, fnames in sorted(os.walk(tgt_dir, followlinks=True)):
                for fname in fnames:
                    path = os.path.join(root, fname)
                    item = (path, tgt_class)
                    self.samples.append(item)
        self.num_examples = len(self.samples)
        self.indices = np.arange(self.num_examples)
        self.num = self.__len__()
        print("Loaded the dataset from {}. It contains {} samples.".format(root_dir, self.num))
        self.transform = transform
  
    def __len__(self):
        return self.num_examples
  
    def __getitem__(self, index):
        index = self.indices[index]
        sample = self.samples[index]
        latents = np.load(sample[0])  # expected shape: (num_aug, num_tokens) int64
        latents = torch.from_numpy(latents).long()  # (num_aug, num_tokens), token IDs must be int64
        latents = latents.unsqueeze(0)              # (1, num_aug, num_tokens) to match old shape

        # select one augmentation (same logic as before)
        aug_idx = torch.randint(0, latents.shape[1], (1,)).item()
        latents = latents[:, aug_idx, :]            # (1, num_tokens)

        label = sample[1]
        return latents, label, index