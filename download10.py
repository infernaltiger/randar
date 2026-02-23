import torch
from torchvision import datasets, transforms

transform = transforms.ToTensor()

print("1")
# Загрузка обучающего и тестового наборов
train_dataset = datasets.CIFAR10(
    root='D:\\inno\\adv_ml_project\\randar\\RandAR\\data\\cifar10',          # куда сохранить
    train=True,
    download=True,          # автоматически скачает, если нет
    transform=transform
)

test_dataset = datasets.CIFAR10(
    root='D:\\inno\\adv_ml_project\\randar\\RandAR\\data\\cifar10',
    train=False,
    download=True,
    transform=transform
)