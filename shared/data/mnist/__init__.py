from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from shared.data.CustomDataLoader import CustomDataLoader


class MNIST_(CustomDataLoader):
    @staticmethod
    def data_loaders(h: dict, transformations: list, dataset_path, batch_size, workers) -> Tuple[
        DataLoader, DataLoader, DataLoader]:
        print("Info: loading MNIST data")
        # transform = transforms.Compose([transforms.ToTensor()] + transformations)
        transform = transforms.Compose([transforms.ToTensor()] + [lambda x: t(x, h) for t in transformations])
        mnist_full = MNIST(dataset_path, train=True, download=True, transform=transform)
        mnist_test = MNIST(dataset_path, train=False, download=True, transform=transform)

        # Split the full training set into training and validation sets
        train_size = int(0.8 * len(mnist_full))  # 80% training, 20% validation
        val_size = len(mnist_full) - train_size
        mnist_train, mnist_val = torch.utils.data.random_split(mnist_full, [train_size, val_size])

        train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(mnist_val, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader
