from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

def get_datasets(data_root="./data", val_frac=0.1, seed=42):
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    train_full = datasets.CIFAR10(root=data_root, train=True, download=True, transform=train_tf)
    testset    = datasets.CIFAR10(root=data_root, train=False, download=True, transform=test_tf)
    val_size = int(len(train_full) * val_frac)
    train_size = len(train_full) - val_size
    g = torch.Generator().manual_seed(seed)
    trainset, valset = random_split(train_full, [train_size, val_size], generator=g)
    return trainset, valset, testset

def make_loaders(trainset, valset, testset, batch_size=128, num_workers=2):
    tr = DataLoader(trainset, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    va = DataLoader(valset,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    te = DataLoader(testset,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return tr, va, te