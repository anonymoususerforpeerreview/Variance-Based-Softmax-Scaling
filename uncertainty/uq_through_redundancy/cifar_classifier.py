from typing import Tuple

import torch
from torch import nn, Tensor
from shared.loss import Loss, LossFactory
from uncertainty.uq_through_redundancy.multi_input_classifier import MultiInputClassifier
from shared.data.dataset_meta import DatasetMeta as M


class NetworkInNetworkClassifier(MultiInputClassifier):
    # Same as MultiInputClassifier, but overwrites the architecture to https://arxiv.org/pdf/1505.00853v2
    # actually, comes from: https://arxiv.org/abs/1312.4400 but i based it on the first link (and they based it on 2nd link)
    def __init__(self, h: dict, in_channels: int, latent_dims: list, latent_strides: list, latent_kernel_sizes: list,
                 latent_padding: list,
                 num_classes: int, loss: Loss, final_activation: str):
        print("NetworkInNetworkClassifier.__init__")
        assert h['dataset'] in ['CIFAR10', 'CIFAR100'], "CifarClassifier only supports CIFAR10 and CIFAR100 datasets."
        super(NetworkInNetworkClassifier, self).__init__(h, in_channels, latent_dims, latent_strides,
                                                         latent_kernel_sizes,
                                                         latent_padding, num_classes, loss, final_activation)

        # over
        self.layers = nn.ModuleList([
            # assumes input shape (batch, in_channels, 32, 32)
            nn.Conv2d(in_channels, 192, kernel_size=5, stride=1, padding=2),  # result: (batch, 192, 32, 32)
            nn.ReLU(),
            nn.Conv2d(192, 160, kernel_size=1, stride=1, padding=0),  # result: (batch, 160, 32, 32)
            nn.ReLU(),
            nn.Conv2d(160, 96, kernel_size=1, stride=1, padding=0),  # result: (batch, 96, 32, 32)
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # result: (batch, 96, 16, 16)
            nn.Dropout(p=0.5),
            nn.Conv2d(96, 192, kernel_size=5, stride=1, padding=2),  # result: (batch, 192, 16, 16)
            nn.ReLU(),
            nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),  # result: (batch, 192, 16, 16)
            nn.ReLU(),
            nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),  # result: (batch, 192, 16, 16)
            nn.ReLU(),

            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),  # result: (batch, 192, 8, 8)
            nn.Dropout(p=0.5),
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),  # result: (batch, 192, 8, 8)
            nn.ReLU(),
            nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),  # result: (batch, 192, 8, 8)
            nn.ReLU(),
            nn.Conv2d(192, num_classes, kernel_size=1, stride=1, padding=0),  # result: (batch, num_classes, 8, 8)

            # Average pooling will take place in the superclass (_forward_single_with_moving_average)
            # nn.ReLU(),
            # nn.AvgPool2d(kernel_size=8, stride=1, padding=0),  # result: (batch, num_classes, 1, 1)
        ])

        # identity function
        self.fc = nn.Identity()  # Not needed as nb_channels already matches num_classes, but was used in MultiInputClassifier

    def _forward_conv_layers(self, x: Tensor) -> Tuple[Tensor, list]:
        """
        Forwards the input through the convolutional layers. (used for performance predictions/inference and uncertainty estimation)
        In: (batch, C, H, W)
        Out: (batch, 512, H', W'), latent_representations
        """
        M.assert_BatchChannelHeightWidth_shape(self.h, x)  # (batch, C, H, W)
        assert self.h['alpha'] == 1, "Distinct Path Regularization not supported for CifarClassifier."
        # No batch norm in this architecture
        for layer in self.layers:
            x = layer(x)
        return x, []

    @staticmethod
    def create_instance(h: dict) -> 'NetworkInNetworkClassifier':
        return NetworkInNetworkClassifier(
            h=h,  # Hyperparameters
            in_channels=M.num_input_channels(h),  # Number of input channels
            latent_dims=[1],
            latent_strides=[],
            latent_kernel_sizes=[],
            latent_padding=[],
            num_classes=M.num_classes(h),  # Number of classes
            loss=LossFactory.create_loss(h),  # Loss function
            final_activation=h['final_activation']
        )


class MCDropoutNetworkInNetworkClassifier(NetworkInNetworkClassifier):
    def __init__(self, h: dict, in_channels: int, latent_dims: list, latent_strides: list, latent_kernel_sizes: list,
                 latent_padding: list, num_classes: int, loss: Loss, final_activation: str, dropout_rate: float):
        super(MCDropoutNetworkInNetworkClassifier, self).__init__(h, in_channels, latent_dims, latent_strides,
                                                                  latent_kernel_sizes,
                                                                  latent_padding, num_classes, loss, final_activation)
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(p=dropout_rate)
        self.train()  # Set model to training mode ALWAYS (for dropout to remain active)

    def _forward_conv_layers(self, x: Tensor) -> Tuple[Tensor, list]:
        """
        Forwards the input through the convolutional layers with dropout. (used for performance predictions/inference and uncertainty estimation)
        In: (batch, C, H, W)
        Out: (batch, 512, H', W'), latent_representations
        """
        M.assert_BatchChannelHeightWidth_shape(self.h, x)  # (batch, C, H, W)
        assert self.h['alpha'] == 1, "Distinct Path Regularization not supported for Monte Carlo Dropout."

        for layer in self.layers:
            # skip the one dropout layer part of the architecture
            if isinstance(layer, nn.Dropout):
                continue

            x = layer(x)
            if isinstance(layer, nn.ReLU):
                x = self.dropout(x)  # Apply dropout after each relu

        return x, []

    @staticmethod
    def create_instance(h: dict) -> 'MCDropoutNetworkInNetworkClassifier':
        return MCDropoutNetworkInNetworkClassifier(
            h=h,  # Hyperparameters
            in_channels=M.num_input_channels(h),  # Number of input channels
            latent_dims=[1],  # Random number just so that \
            # `self.fc = nn.Linear(latent_dims[-1], num_classes)`  works in super class, although it is not used
            latent_strides=[],
            latent_kernel_sizes=[],
            latent_padding=[],
            num_classes=M.num_classes(h),  # Number of classes
            loss=LossFactory.create_loss(h),  # Loss function
            final_activation=h['final_activation'],
            dropout_rate=h.get('dropout_rate', 0.5)  # Dropout rate
        )

    def eval(self):
        self.train()  # Set model to training mode ALWAYS (for dropout to remain active)


if __name__ == "__main__":
    batch = torch.randn(32, 3, 32, 32)
    res = nn.Conv2d(in_channels=3, out_channels=192,
                    kernel_size=5,
                    stride=1, padding=2)(batch)
    print(res.shape)  # torch.Size([32, 192, 32, 32])
