from typing import Tuple

from torch import Tensor
from torch import nn

from shared.data.dataset_meta import DatasetMeta as M
from shared.loss import Loss, LossFactory
from uncertainty.uq_through_redundancy.multi_input_classifier import MultiInputClassifier


class MCDropoutClassifier(MultiInputClassifier):
    def __init__(self, h: dict, in_channels: int, latent_dims: list, latent_strides: list, latent_kernel_sizes: list,
                 latent_padding: list, num_classes: int, loss: Loss, final_activation: str, dropout_rate: float):
        super(MCDropoutClassifier, self).__init__(h, in_channels, latent_dims, latent_strides,
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
            x = layer(x)
            if isinstance(layer, self.batch_norm):
                x = nn.functional.relu(x)
                x = self.dropout(x)  # Apply dropout after each relu

        return x, []

    @staticmethod
    def create_instance(h: dict) -> 'MCDropoutClassifier':
        return MCDropoutClassifier(
            h=h,  # Hyperparameters
            in_channels=M.num_input_channels(h),  # Number of input channels
            latent_dims=h['latent_dims'],
            latent_strides=h['latent_strides'],
            latent_kernel_sizes=h['latent_kernel_sizes'],
            latent_padding=h['latent_padding'],
            num_classes=M.num_classes(h),  # Number of classes
            loss=LossFactory.create_loss(h),  # Loss function
            final_activation=h['final_activation'],
            dropout_rate=h.get('dropout_rate', 0.5)  # Dropout rate
        )

    def eval(self):
        self.train()  # Set model to training mode ALWAYS (for dropout to remain active)
