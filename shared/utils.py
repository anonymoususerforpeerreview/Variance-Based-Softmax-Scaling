from typing import Tuple

import torch
import numpy as np
import os
import json
from shared.data.dataset_meta import DatasetMeta as M

from torch import Tensor


def create_path_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def set_seed(seed):
    if seed == -1:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


class Sanity:
    @staticmethod
    def assert_softmax_is_applied(probs):
        """
        In: (batch, num_classes)
        """
        assert probs.dim() == 2, f"Expected 2 dimensions, got {probs.dim()}."

        # assert softmax is already applied
        assert torch.allclose(probs.sum(dim=1),
                              torch.ones(probs.shape[0], device=probs.device)), \
            f"Expected the sum of the probabilities to be 1, got {probs.sum(dim=1)}."


def get_train_id(h: dict):
    alpha = f"alpha={h['alpha']}" if not (h['alpha'] == '1') else ""
    run_name = f"Tr act={h['final_activation']} T={h['redundancy_method']} nb_v={h['num_views']} {h['run_name']} {alpha} {json.dumps(h['redundancy_method_params'])}"  # Tr for train
    return run_name


def get_analysis_id(h: dict):
    run_name = (
        f"A {h['model_name']} {h['UQ_predictor']} {json.dumps(h['predictor_params'])} act={h['final_activation']} "
        f"T={h['redundancy_method']} nb_v={h['num_views']} {json.dumps(h['redundancy_method_params'])}_")
    # A for analysis
    # !! Final underscore is necessary in case redundancy_method_params is empty as to not leave a space at the end
    # This gives issues when saving under that name on Windows.
    return run_name


class ReshapeUtil:  # Reshape Utils
    @staticmethod
    def permute_channels_to_last_dim(self, x: Tensor) -> Tensor:
        """
        Permute the channels to the last dimension.
        In: (batch, c, h, w) or (batch, c, l)
        Out: (batch, h, w, c) or (batch, l, c)
        """
        M.assert_BatchChannelHeightWidth_shape(self.h, x)  # (batch, c, h, w) or (batch, c, l)

        # Check if the input is 2D or 1D
        if x.dim() == 4:  # 2D case
            return x.permute(0, 2, 3, 1).contiguous()
        elif x.dim() == 3:  # 1D case
            return x.permute(0, 2, 1).contiguous()
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")

    @staticmethod
    def flatten_spatial_dimensions(x: Tensor) -> Tensor:
        """
        Flatten the spatial dimensions.
        In: (batch, h, w, c) or (batch, l, c)
        Out: (batch * h * w, c) or (batch * l, c)
        """

        if x.dim() == 4:  # 2D case
            batch, h, w, c = x.size()
            return x.view(batch * h * w, c)
        elif x.dim() == 3:  # 1D case
            batch, l, c = x.size()
            return x.view(batch * l, c)
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")

    @staticmethod
    def reshape_views_part_of_batch(h: dict, x: Tensor) -> Tuple[Tensor, int, int]:
        """
        Reshape the views part of the batch to a single batch.
        In: (batch, nb_views, C, H, W) or (batch, nb_views, C, L)
        Out: (batch * nb_views, C, H, W) or (batch * nb_views, C, L)
        """
        if M.num_dims(h) == 2:  # Image data
            batch, nb_views, c, h, w = x.shape
            x = x.view(batch * nb_views, c, h, w)  # (batch * nb_views, C, H, W)
        elif M.num_dims(h) == 1:  # Sequence data
            batch, nb_views, c, l = x.shape
            x = x.view(batch * nb_views, c, l)  # (batch * nb_views, C, L)
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")
        return x, batch, nb_views

    @staticmethod
    def unflatten_spatial_dimensions(logits: Tensor, original_z_shape: Tuple[int, ...]) -> Tensor:
        """
        Unflatten the spatial dimensions.
        In: logits: (batch * h * w, nb_classes) or (batch * l, nb_classes), original_shape: (batch, h, w, channels) or (batch, l, channels)
        Out: (batch, h, w, nb_classes) or (batch, l, nb_classes)
        where nb_classes != channels
        """
        b = original_z_shape[0]
        h_w_shape = original_z_shape[1:-1]  # (h', w') or (l')
        nb_classes = logits.size(-1)
        logits = logits.view(b, *h_w_shape, nb_classes)  # (batch, h', w', nb_classes) or (batch, l', nb_classes)
        return logits

    @staticmethod
    def collapse_views_and_spatial(h: dict, logits: Tensor) -> Tensor:
        # In: (batch, nb_views, H', W', num_classes) or (batch, nb_views, L', num_classes)
        # Out: (batch, num_classes)
        M.assert_BatchViewChannelHeightWidth_shape(h, logits)
        spatial_dims = tuple(range(2, len(logits.shape) - 1))  # (2, 3) or (2)
        logits = logits.mean(dim=spatial_dims)  # (batch, nb_views, num_classes)
        logits = logits.mean(dim=1)  # (batch, num_classes)

        M.assert_output_BatchNumClasses(h, logits)
        return logits

    @staticmethod
    def swap_spatial_to_end(h: dict, logits: Tensor) -> Tensor:
        """
        Swap the spatial and channel dimensions.
        In:
            (batch, H, W, nb_classes) or (batch, L, nb_classes) or

        Out:
            (batch, nb_classes, H, W) or (batch, nb_classes, L) or
        """

        if M.is_1d_signal(h):
            assert logits.dim() == 3, f"Expected 3 dimensions, got {logits.dim()}."
            logits = logits.permute(0, 2, 1)  # (b*nb_views, num_classes, L')
        else:
            assert logits.dim() == 4, f"Expected 4 dimensions, got {logits.dim()}."
            logits = logits.permute(0, 3, 1, 2)  # (b*nb_views, num_classes, H', W')
        return logits

    @staticmethod
    def swap_nb_classes_to_end(h: dict, logits: Tensor) -> Tensor:
        """
        Swap the nb_classes and channel dimensions.
        In:
            (batch, nb_classes, H, W) or (batch, nb_classes, L) or

        Out:
            (batch, H, W, nb_classes) or (batch, L, nb_classes) or
        """

        if M.is_1d_signal(h):
            assert logits.dim() == 3, f"Expected 3 dimensions, got {logits.dim()}."
            logits = logits.permute(0, 2, 1)
        else:
            assert logits.dim() == 4, f"Expected 4 dimensions, got {logits.dim()}."
            logits = logits.permute(0, 2, 3, 1)
        return logits
