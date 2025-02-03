from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100

from shared.data.CustomDataLoader import CustomDataLoader
from shared.data.noise import NoiseFactory
from shared.data.transformations import T


# Params cifar: https://arxiv.org/pdf/1312.4400, https://arxiv.org/pdf/1505.00853v2
# orig paper: 200 epochs, weight decay. no augmentations
# follow up paper: batch 128. they do use augmentations


class CustomTransform:
    # A wrapper class for transformations so that h can be passed to the transformation functions
    def __init__(self, transformations, h):
        self.transformations = transformations
        self.h = h

    def __call__(self, x):
        for t in self.transformations:
            x = t(x, self.h)
        return x


class CIFAR10_(CustomDataLoader):
    @staticmethod
    def data_loaders(h: dict, transformations: list, dataset_path, batch_size, workers) -> Tuple[
        DataLoader, DataLoader, DataLoader]:
        print("Info: loading CIFAR-10 data")

        transform = transforms.Compose([transforms.ToTensor(), CustomTransform(transformations, h)])
        cifar10_full = CIFAR10(dataset_path, train=True, download=True, transform=transform)
        cifar10_test = CIFAR10(dataset_path, train=False, download=True, transform=transform)

        train_size = int(0.8 * len(cifar10_full))  # 80% training, 20% validation
        val_size = len(cifar10_full) - train_size
        cifar10_train, cifar10_val = torch.utils.data.random_split(cifar10_full, [train_size, val_size])

        train_loader = DataLoader(cifar10_train, batch_size=batch_size, shuffle=True, num_workers=workers, persistent_workers=True)
        val_loader = DataLoader(cifar10_val, batch_size=batch_size, shuffle=False, num_workers=workers, persistent_workers=True)
        test_loader = DataLoader(cifar10_test, batch_size=batch_size, shuffle=False, num_workers=workers, persistent_workers=True)

        return train_loader, val_loader, test_loader


class CIFAR100_(CustomDataLoader):
    @staticmethod
    def data_loaders(h: dict, transformations: list, dataset_path, batch_size, workers) -> Tuple[
        DataLoader, DataLoader, DataLoader]:
        print("Info: loading CIFAR-100 data")

        transform = transforms.Compose([transforms.ToTensor(), CustomTransform(transformations, h)])

        cifar100_full = CIFAR100(dataset_path, train=True, download=True, transform=transform)
        cifar100_test = CIFAR100(dataset_path, train=False, download=True, transform=transform)

        train_size = int(0.8 * len(cifar100_full))  # 80% training, 20% validation
        val_size = len(cifar100_full) - train_size
        cifar100_train, cifar100_val = torch.utils.data.random_split(cifar100_full, [train_size, val_size])

        train_loader = DataLoader(cifar100_train, batch_size=batch_size, shuffle=True, num_workers=workers)
        val_loader = DataLoader(cifar100_val, batch_size=batch_size, shuffle=False, num_workers=workers)
        test_loader = DataLoader(cifar100_test, batch_size=batch_size, shuffle=False, num_workers=workers)

        return train_loader, val_loader, test_loader

# Function to show an image
def imshow(img):
    # img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def create_dir_if_not_exists(path: str):
    import os
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torchvision
    import torchvision.transforms as transforms
    from torchvision.datasets import CIFAR10
    from torch.utils.data import DataLoader

    # fix seed
    torch.manual_seed(0)

    h = {
        "batch_size": 4,
        "workers": 2,
        "redundancy_method": "identity",
        "redundancy_method_params": {},
        "num_views": 1,
    }
    transformations = [T.create_multiple_views]
    dataset_path = "./../../../../Smooth-InfoMax/datasets"
    batch_size = 8
    workers = 2

    for noise_type in ["gaussian", "affine_transform", "elastic_distortion"]:
        train_loader, _, test_loader = CIFAR10_.data_loaders(h, transformations, dataset_path, batch_size, workers)

        # Get a batch of images
        data_iter = iter(test_loader)
        images, labels = next(data_iter)  # (batch_size, nb_views, channels, height, width)

        # remove nb_views dim (its 1)
        import tikzplotlib

        for i in range(images.size(0)):
            fig, axes = plt.subplots(1, 6, figsize=(20, 5))
            for j, noise_factor in enumerate(np.linspace(0, 1, 6)):
                noisy_image = NoiseFactory.apply_noise(images[i].unsqueeze(0), noise_type, noise_factor)
                noisy_image = noisy_image.squeeze(0)  # remove batch dimension
                noisy_image = noisy_image.squeeze(0)  # remove nb_views dimension
                # images = images.squeeze(1)  # (batch_size, channels, height, width)
                axes[j].imshow(np.transpose(noisy_image.numpy(), (1, 2, 0)))
                axes[j].set_title(f"Noise: {noise_factor:.2f}")
                axes[j].axis("off")

            plt.tight_layout()
            create_dir_if_not_exists(f"noisy_images_{i}_{noise_type}")
            fig.savefig(f"noisy_images_{i}_{noise_type}/img.pdf")
            tikzplotlib.save(f"noisy_images_{i}_{noise_type}/img.tex")
            plt.show()
            plt.close()

            # 7 is frog