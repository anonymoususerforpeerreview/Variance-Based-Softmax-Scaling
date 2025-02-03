from typing import Tuple, Union

import torch
from torch import Tensor
from torch import nn

from shared.abstract_model import ANN
from shared.data.dataset_meta import DatasetMeta as M
from shared.loss import Loss, LossFactory
from shared.utils import ReshapeUtil as RU


class MultiInputClassifier(ANN):
    def __init__(self, h: dict, in_channels: int, latent_dims: list, latent_strides: list, latent_kernel_sizes: list,
                 latent_padding: list,
                 num_classes: int, loss: Loss, final_activation: str):
        super(MultiInputClassifier, self).__init__()

        assert callable(loss), f"Loss must be a callable function, got {loss}."

        self.h = h
        self.latent_dims = latent_dims
        self.loss: Loss = loss

        self.conv, self.batch_norm, self.avg_pool = self._get_conv_batchnorm_avgpool(h)

        # Encoder layers. Each layer will be a reparameterization layer
        layers = []
        for dim, stride, kernel_size, padding in zip(latent_dims, latent_strides, latent_kernel_sizes, latent_padding):
            layers.append(self.conv(in_channels, dim, kernel_size, stride, padding=padding))
            layers.append(self.batch_norm(dim))
            in_channels = dim

        self.layers = nn.ModuleList(layers)
        self.fc = nn.Linear(latent_dims[-1], num_classes)

        supported_activations = ['identity', 'tanh', 'sigmoid', 'softplus', 'exponential', 'relu']
        assert final_activation in supported_activations, \
            f"Final activation must be {supported_activations}, got {final_activation}."
        self.final_activation = self._get_final_activation(final_activation)

    def forward(self, x: Tensor) -> Tuple[Tensor, list]:
        """
        Used for performance predictions. Returns logits (softmax not applied) that are averaged over time and latent representations.
        In: (batch, nb_views, C, H, W)
        Out: (batch, nb_views, num_classes), latent_representations
        """
        M.assert_BatchViewChannelHeightWidth_shape(self.h, x)  # (batch, nb_views, C, H, W)
        return self._forward_multiple(x)

    def _forward_multiple(self, x) -> Tuple[Tensor, list]:
        """Returns the predictions and latent representations of all views of the input.
        In: (batch, nb_views, C, H, W)
        Out: (batch, nb_views, num_classes), latent_representations
        """

        # (batch, nb_views, C, H, W)
        M.assert_BatchViewChannelHeightWidth_shape(self.h, x)

        # (batch * nb_views, C, H, W) or (batch * nb_views, C, L)
        x, batch, nb_views = RU.reshape_views_part_of_batch(self.h, x)

        logits, latent_representations = self._forward_single(x)  # (batch * nb_views, num_classes)
        logits = logits.view(batch, nb_views, M.num_classes(self.h))  # (batch, nb_views, num_classes)

        # Reshape latent representations
        latent_representations = [l.view(batch, nb_views, *l.shape[1:]) for l in latent_representations]
        return logits, latent_representations

    def _forward_conv_layers(self, x: Tensor) -> Tuple[Tensor, list]:
        """
        Forwards the input through the convolutional layers. (used for performance predictions/inference and uncertainty estimation)
        In: (batch, C, H, W)
        Out: (batch, 512, H', W'), latent_representations
        """
        M.assert_BatchChannelHeightWidth_shape(self.h, x)  # (batch, C, H, W)
        latent_representations = []
        if self.h['alpha'] == 1:
            for layer in self.layers:
                x = layer(x)
                if isinstance(layer, self.batch_norm):
                    x = nn.functional.relu(x)
        else:
            for layer in self.layers:
                x = layer(x)
                if isinstance(layer, self.batch_norm):
                    x = nn.functional.relu(x)
                latent_representations.append(x)

        return x, latent_representations

    def _forward_single(self, x) -> Tuple[Tensor, list]:
        """
        Used for inference (used for performance predictions). Returns logits (softmax not applied) that are averaged over time.
        In: (batch, C, H, W)
        Out: (batch, num_classes), latent_representations
        """
        # (batch, H', W', num_classes) or (batch, L', num_classes)
        logits, latent_representations = self._forward_single_with_moving_average(x, kernel_size=0)

        # remove spatial dimensions as they are 1 (H' = 1, W' = 1)
        logits = logits.view(logits.size(0), -1)  # (batch, num_classes)

        return logits, latent_representations

    def _forward_single_with_moving_average(self, x, kernel_size: Union[int, Tuple] = 0, stride=1) -> Tuple[Tensor, list]:
        """
        Used for uncertainty estimation; returns logits (softmax not applied) computed on sub-images of the input.
        In: (batch, C, H, W) or (batch, C, L)
        Out: (batch, H', W', num_classes) or (batch, L', num_classes), latent_representations
        """
        z, latent_representations = self._forward_conv_layers(x)

        # average pooling with kernel_size over time (sliding window)

        # For example, if you use a kernel_size of (2, 2) and a stride of (2, 2) for a 2D image,
        # the spatial dimensions will be halved. The resulting shape will be (batch, c, h/2, w/2)
        # if the input shape was (batch, c, h, w). Currently, kernel_size = stride.
        if kernel_size == 0:  # No average pooling
            kernel_size = z.size()[2:]  # (H, W) or (L) for sequences
        z = self.avg_pool(z, kernel_size=kernel_size, stride=stride)  # (batch, c, h', w') or (batch, c, l')

        # (batch, c, h', w') -> (batch, h', w', c)
        z = RU.permute_channels_to_last_dim(self, z)
        z_shape = z.size()  # (batch, h_prime, w_prime, c) or (batch, l_prime, c)

        # Flatten the spatial dimensions and predict on the subimages -> (batch * h' * w', c) or (batch * l', c)
        z_flattened_spatial = RU.flatten_spatial_dimensions(z)

        # Apply the fully connected layer
        logits = self.fc(z_flattened_spatial)  # (batch * h' * w', num_classes) or (batch * l', num_classes)

        # Reshape the output to the original shape (from (b * l', num_classes) to (batch, l', num_classes))
        logits = RU.unflatten_spatial_dimensions(logits, z_shape)

        # Apply the final activation
        logits = self.final_activation(logits)

        return logits, latent_representations

    def _get_conv_batchnorm_avgpool(self, h: dict):
        if M.num_dims(h) == 2:  # Images
            conv = nn.Conv2d
            batch_norm = nn.BatchNorm2d
            avg_pool = nn.functional.avg_pool2d
        elif M.num_dims(h) == 1:  # Sequences
            conv = nn.Conv1d
            batch_norm = nn.BatchNorm1d
            avg_pool = nn.functional.avg_pool1d
        else:
            raise ValueError(f"Unsupported number of dimensions: {M.num_dims(h)}")
        return conv, batch_norm, avg_pool

    def _get_final_activation(self, final_activation: str):
        if final_activation == 'identity':
            return lambda x: x
        elif final_activation == 'tanh':
            return torch.tanh
        elif final_activation == 'sigmoid':
            return torch.sigmoid
        elif final_activation == 'softplus':
            return torch.nn.functional.softplus
        elif final_activation == 'exponential':
            return torch.nn.functional
        elif final_activation == 'relu':
            return torch.nn.functional.relu
        else:
            raise ValueError(f"Unsupported final activation: {final_activation}")

    @staticmethod
    def create_instance(h: dict) -> 'MultiInputClassifier':
        return MultiInputClassifier(
            h=h,  # Hyperparameters
            in_channels=M.num_input_channels(h),  # Number of input channels
            latent_dims=h['latent_dims'],
            latent_strides=h['latent_strides'],
            latent_kernel_sizes=h['latent_kernel_sizes'],
            latent_padding=h['latent_padding'],
            num_classes=M.num_classes(h),  # Number of classes
            loss=LossFactory.create_loss(h),  # Loss function
            final_activation=h['final_activation']
        )


if __name__ == '__main__':
    # Test on Radio dataset
    # h = {
    #     'dataset': 'RADIO',
    #     'num_views': 1,
    #     'latent_dims': [512, 512, 512, 512, 512],
    #     'latent_strides': [5, 4, 1, 1, 1],
    #     'latent_kernel_sizes': [10, 8, 4, 4, 4],
    #     'latent_padding': [2, 2, 2, 2, 1],
    #     'alpha': 0,
    #     'redundancy_method_uncertainty_kernel_size': 20,
    # }
    # c = MultiInputClassifier.create_instance(h)
    # # print(c)
    #
    # # 1880 = 2080 - 200
    # # if 1880 -> l' is 94
    # batch = torch.randn(32, h['num_views'], 2, 1880)  # (batch, nb_views, C, L)
    # out, latent = c.forward(batch)
    # print(out.shape)

    #  --latent_dims [128,128,128,128,128] --latent_strides [5,4,2,2,2] --latent_kernel_sizes [10,8,4,4,4] --latent_padding [2,2,2,2,1] --dataset_path ./../Smooth-InfoMax\datasets --batch_size 8
    # Test on LibriSpeech dataset
    # h = {
    #     'dataset': 'LIBRISPEECH_GIM_SUBSET',
    #     'redundancy_method': 'random_crop',
    #     'redundancy_method_params': {'crop_size': 20480},
    #
    #     'num_views': 1,
    #     'batch_size': 8,
    #
    #     'latent_dims': [512, 512, 512, 512, 512],
    #     'latent_strides': [5, 4, 2, 2, 2],
    #     'latent_kernel_sizes': [10, 8, 4, 4, 4],
    #     'latent_padding': [2, 2, 2, 2, 1],
    #     'alpha': 0,
    # }
    #
    # c = MultiInputClassifier.create_instance(h)
    # batch = torch.randn(h['batch_size'], 1, 20480)  # (batch, C, L)
    #
    # out, latent = c._forward_conv_layers(batch)
    # print(out.shape)  # (8, 512, 128) so 160-fold reduction in length

    #--latent_dims
    # "[512, 512, 512, 512, 512]"
    # --latent_strides
    # "[5, 4, 1, 1, 1]"
    # --latent_kernel_sizes
    # "[10, 8, 4, 4, 4]"
    # --latent_padding
    # [2,2,2,2,1]

    latent_dims = [512, 512, 512, 512, 512]
    latent_strides = [5, 4, 1, 1, 1]
    latent_kernel_sizes = [10, 8, 4, 4, 4]
    latent_padding = [2, 2, 2, 2, 1]

    input_tensor = torch.randn(32, 2, 2080 - 500)  # (batch_size, channels, length)

    in_channels = 2
    x = input_tensor
    for i, (dim, stride, kernel_size, padding) in enumerate(
            zip(latent_dims, latent_strides, latent_kernel_sizes, latent_padding)):
        conv_layer = nn.Conv1d(in_channels, dim, kernel_size, stride, padding=padding)
        x = conv_layer(x)
        x = nn.ReLU()(x)
        print(f"Output of layer {i + 1}: {x.shape}")
        in_channels = dim
