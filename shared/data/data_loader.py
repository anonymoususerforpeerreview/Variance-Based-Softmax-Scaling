from typing import Tuple

from torch.utils.data import DataLoader

from shared.data.cifar import CIFAR10_, CIFAR100_
from shared.data.librispeech import LIBRISPEECH
from shared.data.librispeech_gim_subset import LIBRISPEECH_GIM_SUBSET
from shared.data.mnist import MNIST_
from shared.data.radio import RADIO
from shared.data.toyregression.toy_regression import TOYREGRESSION
from shared.data.transformations import T
from uncertainty.analysis.visual import Visual

# Add a flag to ensure show_transformations is called only once
show_transformations_called = False


def data_loaders(h: dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    global show_transformations_called
    path, b, workers = h['dataset_path'], h['batch_size'], h['num_workers']

    transformations = [T.create_multiple_views]

    if h['dataset'] == 'MNIST':
        if not show_transformations_called:
            Visual.show_transformations(h, dataset_path=h['dataset_path'],
                                        transformations=[T.retrieve_transforms(h)],
                                        num_images=2)
            show_transformations_called = True
        return MNIST_.data_loaders(h, transformations, path, b, workers)
    elif h['dataset'] == 'CIFAR10':
        return CIFAR10_.data_loaders(h, transformations, path, b, workers)
    elif h['dataset'] == 'CIFAR100':
        return CIFAR100_.data_loaders(h, transformations, path, b, workers)
    elif h['dataset'] == 'RADIO':
        return RADIO.data_loaders(h, transformations, path, b, workers)
    elif h['dataset'] == 'LIBRISPEECH':
        return LIBRISPEECH.data_loaders(h, transformations, path, b, workers)
    elif h['dataset'] == 'LIBRISPEECH_GIM_SUBSET':
        return LIBRISPEECH_GIM_SUBSET.data_loaders(h, transformations, path, b, workers)
    elif h['dataset'] == 'TOYREGRESSION':
        return TOYREGRESSION.data_loaders(h, transformations, path, b, workers)


# given dataloader
# for i, (x, y) in enumerate(train_loader):
#   x.shape = (batch, nb_views, C, H, W)

