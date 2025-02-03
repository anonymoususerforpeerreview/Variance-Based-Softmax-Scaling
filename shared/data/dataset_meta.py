from typing import Tuple, Optional

import torch
from torch import Tensor


class DatasetMeta:
    DATASETS_1D = ['RADIO', 'LIBRISPEECH', 'LIBRISPEECH_GIM_SUBSET']  # 1D signals
    DATASETS_2D = ['MNIST', 'CIFAR10', 'CIFAR100']  # 2D images

    @staticmethod
    def is_vision_dataset(h: dict) -> bool:
        vision = h['dataset'] in DatasetMeta.DATASETS_2D
        return vision

    @staticmethod
    def num_dims(h: dict):
        dataset_name = h['dataset']
        if dataset_name in DatasetMeta.DATASETS_2D:
            return 2
        elif dataset_name in DatasetMeta.DATASETS_1D:
            return 1
        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}")

    @staticmethod
    def num_input_channels(h: dict):
        dataset_name = h['dataset']
        if dataset_name == 'MNIST':
            return 1
        elif dataset_name in ['CIFAR10', 'CIFAR100']:
            return 3
        elif dataset_name == 'RADIO':
            return 2
        elif dataset_name in ['LIBRISPEECH', 'LIBRISPEECH_GIM_SUBSET']:
            return 1
        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}")

    @staticmethod
    def num_classes(h: dict):
        if h['use_only_k_labels'] > 0:
            return int(h['use_only_k_labels'])

        dataset_name = h['dataset']
        if dataset_name == 'MNIST':
            return 10
        elif dataset_name == 'CIFAR10':
            return 10
        elif dataset_name == 'CIFAR100':
            return 100
        elif dataset_name == 'RADIO':
            return 20
        elif dataset_name == 'LIBRISPEECH':
            # nb of speakers
            return 10_000  # 2484
        elif dataset_name == 'LIBRISPEECH_GIM_SUBSET':
            # nb of speakers
            return 251
        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}")

    @staticmethod
    def assert_BatchViewChannelHeightWidth_shape(h: dict, x: torch.Tensor):
        num_dims = None
        target_shape = None
        if h['dataset'] in DatasetMeta.DATASETS_2D:
            num_dims = 5
            target_shape = "(batch, nb_views, C, H, W)"
        elif h['dataset'] in DatasetMeta.DATASETS_1D:
            num_dims = 4
            target_shape = "(batch, nb_views, C, Length)"
        assert x.dim() == num_dims, f"Input shape is {x.shape}, expected {num_dims} dimensions. {target_shape}"

    @staticmethod
    def assert_BatchChannelHeightWidth_shape(h: dict, x: torch.Tensor):
        num_dims = None
        target_shape = None
        if h['dataset'] in DatasetMeta.DATASETS_2D:
            num_dims = 4
            target_shape = "(batch, C, H, W)"
        elif h['dataset'] in DatasetMeta.DATASETS_1D:
            num_dims = 3
            target_shape = "(batch, C, Length)"
        assert x.dim() == num_dims, f"Input shape is {x.shape}, expected {num_dims} dimensions. {target_shape}"

    @staticmethod
    def assert_output_BatchNumClasses(h: dict, output: torch.Tensor):
        assert len(output.shape) == 2, f"Expected output to have 2 dimensions, got {len(output.shape)}"
        assert output.shape[1] == DatasetMeta.num_classes(h), \
            f"Expected output to have {DatasetMeta.num_classes(h)} classes, got {output.shape[1]}"

    @staticmethod
    def is_1d_signal(h: dict) -> bool:
        return DatasetMeta.num_dims(h) == 1

    @staticmethod
    def display_samples(h: dict) -> bool:
        # whether to display the data samples from the dataset (e.g. for radio, we don't want to display the samples, but for mnist we do)
        return h['dataset'] in DatasetMeta.DATASETS_2D


    ############################################################################################################
    # DECORATORS #
    @staticmethod
    def assert_batch_view_channel_height_width_input(func):
        # WARNING: ensure that it is called from a class instance that contains self.h.
        # the second argument is expeccted to be the input x
        # e.g. def forward(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        def wrapper(*args, **kwargs) -> Tuple[Tensor, Optional[Tensor]]:
            # args[0] is self, args[1] is x
            h = args[0].h
            x = args[1]
            DatasetMeta.assert_BatchViewChannelHeightWidth_shape(h, x)

            return func(*args, **kwargs)

        return wrapper

    @staticmethod
    def assert_batch_num_classes_output(func):
        # Warning: same as assert_batch_view_channel_height_width_input
        def wrapper(*args, **kwargs) -> Tuple[Tensor, Optional[Tensor]]:
            # args[0] is self, args[1] is x
            output, std = func(*args, **kwargs)

            DatasetMeta.assert_output_BatchNumClasses(args[0].h, output)
            # assert len(output.shape) == 2, f"Expected output to have 2 dimensions, got {len(output.shape)}"
            # h = args[0].h  # self.h
            # assert output.shape[1] == DatasetMeta.num_classes(h), \
            #     f"Expected output to have {DatasetMeta.num_classes(h)} classes, got {output.shape[1]}"
            return output, std

        return wrapper

    ############################################################################################################
    # END DECORATORS #
