# RandAR/dataset/builder.py
import os
from torchvision import datasets, transforms

def build_dataset(is_train, args, transform):
    if args.dataset == "imagenet":
        from .imagenet import ImageTarDataset
        root = os.path.join(args.data_path, "train.tar" if is_train else "val.tar")
        dataset = ImageTarDataset(root, return_labels=True, transform=transform)
        dataset.nb_classes = 1000

    elif args.dataset == "cifar10":
        # IMPORTANT: This branch is for latent extraction only,
        # because training in RandAR usually uses --dataset latent
        # after you have extracted .npy codes.
        from .cifar10 import CIFAR10WithIndex

        dataset = CIFAR10WithIndex(
            root=args.data_path,
            train=is_train,
            transform=transform,
            download=False,
        )
        dataset.nb_classes = 10

    elif args.dataset == "latent":
        # This must remain unchanged: RandAR training reads .npy codes with INatLatentDataset.
        from .imagenet import INatLatentDataset
        dataset = INatLatentDataset(root_dir=args.data_path, transform=transform)

    else:
        raise NotImplementedError

    return dataset
