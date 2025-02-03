import copy
from abc import ABC, abstractmethod
from typing import Tuple, Union, Optional

import torch
from torch import Tensor
from torch import nn, optim

from shared.abstract_model import ANN
from shared.data import filter_iterator
from shared.data.data_loader import data_loaders
from shared.data.dataset_meta import DatasetMeta as M
from shared.loss import softplus_based_softmax
from uncertainty.uq_through_redundancy.cifar_classifier import MCDropoutNetworkInNetworkClassifier
from uncertainty.uq_through_redundancy.ensemble_classifier import EnsembleClassifier
from uncertainty.uq_through_redundancy.mc_dropout_classifier import MCDropoutClassifier
from uncertainty.uq_through_redundancy.multi_input_classifier import MultiInputClassifier, RU


class UQPredictorFactory:
    SUPPORTED_PREDICTOR_TYPES = ['StdBasedSoftmaxRelaxation', 'StdBasedSoftmaxRelaxationLearnedParams', 'Naive',
                                 'AvgSoftmax', 'EnsembleAvgSoftmax', 'EnsembleStdBasedSoftmaxRelaxation',
                                 'EnsembleStdBasedSoftmaxRelaxationLearnedParams', 'MCDropout']

    @staticmethod
    def create_instance(h: dict, model: ANN, predictor_type: str, predictor_params: dict) \
            -> 'UQPredictor':
        assert predictor_type in UQPredictorFactory.SUPPORTED_PREDICTOR_TYPES, \
            f"Unknown predictor type: {predictor_type}"

        if predictor_type == 'StdBasedSoftmaxRelaxation':
            assert 'tau_shift' in predictor_params, "tau_shift is required for StdBasedSoftmaxRelaxationPredictor"
            assert 'tau_amplifier' in predictor_params, "tau_amplifier is required for StdBasedSoftmaxRelaxationPredictor"
            assert 'kernel_size' in predictor_params, "kernel_size is required for StdBasedSoftmaxRelaxationPredictor"
            assert 'std_on_representations' in predictor_params, \
                "std_on_representations is required for StdBasedSoftmaxRelaxationPredictor"

            # Assert model is MultiInputClassifier (not EnsembleClassifier)
            assert isinstance(model, MultiInputClassifier), \
                f"When running StdBasedSoftmaxRelaxation, model must be MultiInputClassifier, not {type(model)}"

            return StdBasedSoftmaxRelaxationPredictor(h, model, **predictor_params)  # tau_shift, kernel_size
        elif predictor_type == 'StdBasedSoftmaxRelaxationLearnedParams':
            assert 'kernel_size' in predictor_params, "kernel_size is required for StdBasedSoftmaxRelaxationPredictor"
            assert 'std_on_representations' in predictor_params, \
                "std_on_representations is required for StdBasedSoftmaxRelaxationPredictor"

            # Assert model is MultiInputClassifier (not EnsembleClassifier)
            assert isinstance(model, MultiInputClassifier), \
                f"When running StdBasedSoftmaxRelaxation, model must be MultiInputClassifier, not {type(model)}"

            return StdBasedSoftmaxRelaxationLearnedParamsPredictor(h, model, **predictor_params)
        elif predictor_type == 'Naive':
            assert 'kernel_size' in predictor_params, "kernel_size is required for Naive"
            assert isinstance(model, MultiInputClassifier), \
                f"When running Naive, model must be MultiInputClassifier, not {type(model)}"
            return Naive(h, model, **predictor_params)
        elif predictor_type == 'AvgSoftmax':  # MultiInputClassifier
            return AvgSoftmax(h, model)
        elif predictor_type == 'EnsembleAvgSoftmax':  # EnsembleClassifier
            return EnsembleAvgSoftmax(h, model)
        elif predictor_type == 'EnsembleStdBasedSoftmaxRelaxation':  # EnsembleClassifier
            assert 'tau_shift' in predictor_params, "tau_shift is required for EnsembleStdBasedSoftmaxRelaxation"
            assert 'tau_amplifier' in predictor_params, "tau_amplifier is required for EnsembleStdBasedSoftmaxRelaxation"
            assert 'transformation' in predictor_params, "transformation is required for EnsembleStdBasedSoftmaxRelaxation"
            return StdBasedSoftmaxRelaxationPredictor4Ensemble(h, model, **predictor_params)
        elif predictor_type == 'EnsembleStdBasedSoftmaxRelaxationLearnedParams':  # EnsembleClassifier
            return StdBasedSoftmaxRelaxationPredictor4EnsembleLearnedParams(h, model)
        elif predictor_type == 'MCDropout':
            assert 'num_samples' in predictor_params, "num_samples is required for MCDropout"
            return AvgSoftmaxMCDropout(h, model, **predictor_params)
        else:
            raise ValueError(f"Unknown predictor type: {predictor_type}")


class UQPredictor(ABC):
    """
    A wrapper around the model that allows for uncertainty estimation.
    """

    def __init__(self, h: dict, model: ANN):
        assert h['num_views'] == 1, "UQPredictor is only implemented for num_views=1"
        self.h: dict = h
        self.model: ANN = model

    @abstractmethod
    def forward(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """
        In: (batch, nb_views, C, H, W)
        Out: (batch, num_classes), (batch,)
            --> returns distributions (softmax IS applied) and standard deviation
        """
        pass

    def eval(self):
        self.model.eval()

    @property
    def device(self):
        return self.model.device


def softmax(h: dict, logits: Tensor) -> Tensor:
    """
    In: (batch, num_classes)
    Out: (batch, num_classes)
    """
    if h['loss'] == 'cross_entropy':
        return nn.functional.softmax(logits, dim=-1)
    elif h['loss'] == 'softplus_based_cross_entropy':
        return softplus_based_softmax(logits)
    else:
        raise ValueError(f"Unknown loss: {h['loss']}")


class StdBasedSoftmaxRelaxationPredictor(UQPredictor):  # Standard deviation based softmax relaxation
    # Computes standard deviation between sub-image predictions and uses it to smooth the full prediction.
    def __init__(self, h: dict, model: MultiInputClassifier, tau_shift: Union[float, str], tau_amplifier,
                 kernel_size: Union[int, Tuple], std_on_representations: str):
        super().__init__(h, model)
        self.model: MultiInputClassifier = model  # type hint
        self.kernel_size = kernel_size
        assert (std_on_representations in
                ['logits', 'softmax', 'both', 'logits_relu', 'logits_softplus',
                 'logits_exp']), f"Unknown std_on_representations: {std_on_representations}"
        self.std_on_representations: str = std_on_representations

        if tau_shift in ['auto', 'mean'] or str(tau_shift).startswith('q'):
            tau_shift = self._compute_taw_shift(h, computation=tau_shift)
            print(f"Auto tau_shift: {tau_shift}")

        self.tau_shift: float = tau_shift
        self.tau_amplifier: float = tau_amplifier

    def forward(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Used for uncertainty estimation; returns distributions (softmax IS applied) by smoothing full prediction based
        on standard devitation of sub-image predictions.
        In: (batch, nb_views, C, H, W)
        Out: (batch, num_classes), (batch,)
        """
        M.assert_BatchViewChannelHeightWidth_shape(self.h, x)
        x, batch, nb_views = RU.reshape_views_part_of_batch(self.h, x)

        logits, _ = self.model._forward_single_with_moving_average(
            x, kernel_size=1)  # (b*nb_views, H', W', num_classes)

        tau, standard_deviations = self._compute_tau_and_standard_deviations(
            logits, batch, nb_views)  # (batch, num_classes), (batch,)

        # Compute final softmax
        logits = logits.view(batch, nb_views, *logits.shape[1:])  # (batch, nb_views, H', W', num_classes)
        logits = RU.collapse_views_and_spatial(self.h, logits)  # (batch, num_classes)

        distributions = softmax(self.h, logits / tau)  # (batch, num_classes)
        return distributions, standard_deviations.squeeze()  # (batch, num_classes), (batch,)

    def _compute_tau_and_standard_deviations(self, logits: Tensor, batch: int, nb_views: int) -> Tuple[Tensor, Tensor]:
        # Not used for final softmax activation. Only for standard deviations computation.
        # We compuate standard deviations based on window-averaged logits (as to have enough context),
        # but the final softmax is still applied on the full logits. For some reason, accuracy was different from
        # normal when computing final softmax on avg of already averaged logits.
        # Average pool the logits over time domain
        # put spatial dimensions at the end
        logits_spatial_at_end = RU.swap_spatial_to_end(self.h, logits)  # (batch * nb_views, num_classes, H', W')
        logits_window_pooled = self.model.avg_pool(  # -> (batch * nb_views, num_classes, H', W')
            logits_spatial_at_end, kernel_size=self.kernel_size, stride=1)

        logits_window_pooled = RU.swap_nb_classes_to_end(
            self.h, logits_window_pooled)  # (batch * nb_views, H', W', num_classes)

        logits_window_pooled = logits_window_pooled.view(
            batch, nb_views, *logits_window_pooled.shape[1:])  # (batch, nb_views, H', W', num_classes)
        standard_deviations = self._compute_standard_deviations(logits_window_pooled)  # (batch, 1)
        tau = self._compute_sigma_tilde(standard_deviations)
        return tau, standard_deviations

    def _compute_standard_deviations(self, logits: Tensor) -> Tensor:
        """
        Compute the standard deviation of the logits for each class.
        In: (batch, nb_views, H', W', num_classes)
        Out: (batch, 1)
        """
        spatial_dims = tuple(range(2, len(logits.shape) - 1))  # (2, 3) for (H', W') or (2) for (L',)
        if self.std_on_representations == 'logits_relu':
            # std on ReLU(logits)
            logits = nn.functional.relu(logits)  # +12_000
            standard_deviations = torch.std(logits, dim=spatial_dims)  # (batch, nb_views, num_classes)
            standard_deviations = torch.mean(standard_deviations, dim=-1)  # (batch, nb_views)
        elif self.std_on_representations == 'logits_exp':
            # std on exp(logits)
            logits = torch.clamp(logits, max=20)  # prevent overflow
            logits = torch.exp(logits)
            standard_deviations = torch.std(logits, dim=spatial_dims)
            standard_deviations = torch.mean(standard_deviations, dim=-1)
        elif self.std_on_representations == 'logits_softplus':
            # std on softplus(logits)
            logits = nn.functional.softplus(logits)
            standard_deviations = torch.std(logits, dim=spatial_dims)
            standard_deviations = torch.mean(standard_deviations, dim=-1)
        elif self.std_on_representations in ['logits', 'both']:
            # std on logits
            standard_deviations = torch.std(logits, dim=spatial_dims)  # (batch, nb_views, num_classes)
            standard_deviations = torch.mean(standard_deviations, dim=-1)  # (batch, nb_views)
        else:
            standard_deviations = torch.ones(logits.size(0), logits.size(1), device=logits.device)  # (batch, nb_views)

        if self.std_on_representations in ['softmax', 'both']:
            # std on softmax
            logits2 = softmax(self.h, logits)
            standard_deviations2 = torch.std(logits2, dim=spatial_dims)  # (batch, nb_views, num_classes)
            standard_deviations2 = torch.mean(standard_deviations2, dim=-1)  # (batch, nb_views)
        else:
            standard_deviations2 = torch.ones(logits.size(0), logits.size(1), device=logits.device)  # (batch, nb_views)

        return torch.mean(standard_deviations, dim=-1) * torch.mean(standard_deviations2, dim=-1)  # (batch, 1)

    def _compute_sigma_tilde(self, standard_deviations: Tensor) -> Tensor:
        """ In: (batch, 1) """
        ones = torch.ones_like(standard_deviations)  # (batch, 1)
        tau = self.tau_amplifier * (standard_deviations + self.tau_shift)  # (batch, 1)
        standard_deviations = torch.max(tau, ones)
        return standard_deviations.unsqueeze(-1).expand(-1, M.num_classes(self.h))  # (batch, num_classes)

    def _compute_taw_shift(self, h: dict, computation: str) -> float:
        """Iterate over the entire validation data to find a good tau_shift value."""
        from shared.data.data_loader import data_loaders

        _, val_loader, _ = data_loaders(h)
        standard_deviations = []  # (batch, 1)
        for i, (batch_clean, labels) in filter_iterator(val_loader, h['limit_val_batches'], log_progress=True):
            batch_clean, b_size, nb_views = RU.reshape_views_part_of_batch(self.h, batch_clean)

            logits, _ = self.model._forward_single_with_moving_average(
                batch_clean, self.kernel_size)  # (b*nb_views, H', W', num_classes)
            logits = logits.view(b_size, nb_views, *logits.shape[1:])  # (batch, nb_views, H', W', num_classes)
            stds = self._compute_standard_deviations(logits)  # (batch, 1)
            standard_deviations.extend(stds.squeeze().tolist())

        standard_deviations = torch.tensor(standard_deviations)
        if computation == 'auto':
            return - standard_deviations.mean().item() + 0.5  # scalar
        elif computation == 'mean':
            return - standard_deviations.mean().item()
        elif computation.startswith('q'):
            q: float = float(computation[1:]) / 100.0  # e.g. 'q25' -> 0.25
            return - torch.quantile(standard_deviations, q).item()
        else:
            raise ValueError(f"Unknown tau shift computation: {computation}")


class Naive(UQPredictor):
    # Computes individual predictions (already softmax) for each sub-image and averages them.
    def __init__(self, h: dict, model: MultiInputClassifier, kernel_size: Union[int, Tuple]):
        assert h['method'] == 'UQ_rednd' and isinstance(model, MultiInputClassifier), \
            (f"Naive is only supported for MultiInputClassifier. Got {type(model)} and method {h['method']}. "
             f"A naive UQPredictor for Cifar doesn't exist.")
        super().__init__(h, model)
        self.kernel_size = kernel_size

    @M.assert_batch_view_channel_height_width_input  # (batch, nb_views, C, H, W)
    @M.assert_batch_num_classes_output  # (batch, num_classes)
    def forward(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """
        In: (batch, nb_views, C, H, W)
        Out: (batch, num_classes), None
            The out distributions are SoftMaxed.
        """
        x, batch, nb_views = RU.reshape_views_part_of_batch(self.h, x)

        logits, _ = self.model._forward_single_with_moving_average(
            x, kernel_size=self.kernel_size)  # (b*nb_views, H', W', num_classes)
        logits = logits.view(batch, nb_views, *logits.shape[1:])  # (batch, nb_views, H', W', num_classes)

        distributions = softmax(self.h, logits)  # (batch, nb_views, H', W', num_classes)
        distributions = RU.collapse_views_and_spatial(self.h, distributions)  # (batch, num_classes)

        return distributions, None


class StdBasedSoftmaxRelaxationLearnedParamsPredictor(
    StdBasedSoftmaxRelaxationPredictor):  # Standard deviation based softmax relaxation where tau_shift and tau_amplifier are learned
    def __init__(self, h: dict, model: MultiInputClassifier, kernel_size: Union[int, Tuple],
                 std_on_representations: str):
        super().__init__(h, model, 0.0, 1.0, kernel_size, std_on_representations)
        self._learn_params()

    def _learn_params(self) -> Tuple[float, float]:
        self.tau_shift = nn.Parameter(torch.tensor(self.tau_shift))
        self.tau_amplifier = nn.Parameter(torch.tensor(self.tau_amplifier))

        optimizer = optim.Adam([self.tau_shift, self.tau_amplifier], lr=1e-2)
        criterion = nn.NLLLoss()

        h_temp = copy.deepcopy(self.h)
        h_temp['batch_size'] = 1024
        train_loader, _, _ = data_loaders(h_temp)
        self.model.eval()

        for epoch in range(10):  # Number of epochs can be adjusted
            for batch in train_loader:
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()

                distributions, _ = self.forward(x)  # (batch, num_classes)
                log_distributions = torch.log(distributions)  # Apply log_softmax

                loss = criterion(log_distributions, y)
                loss.backward()
                optimizer.step()

            print(self.tau_amplifier, self.tau_shift, self.tau_amplifier.grad, self.tau_shift.grad)
            print(f"Epoch: {epoch}, Loss: {loss.item()}")

        print(f"Learned tau_shift: {self.tau_shift.item()}, tau_amplifier: {self.tau_amplifier.item()}")

        return self.tau_shift.item(), self.tau_amplifier.item()


class AvgSoftmax(UQPredictor):
    def __init__(self, h: dict, model: MultiInputClassifier):
        assert h['method'] == 'UQ_rednd' and isinstance(model, MultiInputClassifier), \
            f"AvgSoftmax is only supported for MultiInputClassifier. Got {type(model)} and method {h['method']}"
        super().__init__(h, model)
        self.model: MultiInputClassifier = model  # type hint

    @M.assert_batch_view_channel_height_width_input  # (batch, nb_views, C, H, W)
    @M.assert_batch_num_classes_output  # (batch, num_classes)
    def forward(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Typical model prediction; create logits... avg them and apply softmax.
        In: (batch, nb_views, C, H, W)
        Out: (batch, num_classes), None # None is for standard deviation
            The out distributions are SoftMaxed.
        """
        # M.assert_BatchViewChannelHeightWidth_shape(self.h, x)
        logits, _ = self.model.forward(x)  # (batch, nb_views, num_classes)
        distributions = softmax(self.h, logits)  # (batch, nb_views, num_classes)
        distributions = self._collapse_logits(distributions)  # (batch, num_classes)

        return distributions, None

    def _collapse_logits(self, logits: Tensor) -> Tensor:
        return logits.mean(dim=1)  # (batch, num_classes)


class AvgSoftmaxMCDropout(UQPredictor):
    def __init__(self, h: dict, model: Union[MCDropoutClassifier, MCDropoutNetworkInNetworkClassifier],
                 num_samples: int):
        assert h['method'] == 'mc_dropout' and (
                isinstance(model, MCDropoutClassifier) or
                isinstance(model, MCDropoutNetworkInNetworkClassifier)), \
            f"AvgSoftmaxMC_Dropout is only supported for MonteCarloDropoutClassifier. Got {type(model)} and method {h['method']}"
        super().__init__(h, model)
        self.num_samples: int = num_samples

    # Only supported for MonteCarloDropoutClassifier
    def forward(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Typical model prediction; create logits... avg them and apply softmax.
        In: (batch, nb_views, C, H, W) or (batch, nb_views, C, L)
        Out: (batch, num_classes), None # None is for standard deviation
            The out distributions are SoftMaxed.
        """
        M.assert_BatchViewChannelHeightWidth_shape(self.h, x)

        if M.is_1d_signal(self.h):
            # Repeat the input for num_samples
            batch_size, nb_views, C, L = x.shape
            x_repeated = x.unsqueeze(0).repeat(self.num_samples, 1, 1, 1, 1)  # (num_samples, batch, nb_views, C, L)
            x_repeated = x_repeated.view(-1, nb_views, C, L)  # Flatten: (num_samples * batch, nb_views, C, L)
        else:  # 2D
            # Repeat the input for num_samples
            batch_size, nb_views, C, H, W = x.shape
            x_repeated = x.unsqueeze(0).repeat(self.num_samples, 1, 1, 1, 1,
                                               1)  # (num_samples, batch, nb_views, C, H, W)
            x_repeated = x_repeated.view(-1, nb_views, C, H, W)  # Flatten: (num_samples * batch, nb_views, C, H, W)

        # Perform a single batched forward pass
        logits = self.model.forward(x_repeated)[0]  # (num_samples * batch, nb_views, num_classes)

        # Reshape to group predictions by sample
        logits = logits.view(self.num_samples, batch_size, nb_views, -1)  # (num_samples, batch, nb_views, num_classes)

        # Apply softmax across classes
        distributions = softmax(self.h, logits)  # (num_samples, batch, nb_views, num_classes)

        # Average the distributions across samples
        distributions = distributions.mean(dim=0)  # (batch, nb_views, num_classes)

        # Collapse logits if needed
        distributions = self._collapse_logits(distributions)  # (batch, num_classes)

        return distributions, None

    def _collapse_logits(self, logits: Tensor) -> Tensor:
        return logits.mean(dim=1)  # (batch, num_classes)


class EnsembleAvgSoftmax(UQPredictor):
    def __init__(self, h: dict, model: EnsembleClassifier):
        assert h['method'] == 'ensemble' and isinstance(model, EnsembleClassifier), \
            f"EnsembleAvgSoftmax is only supported for EnsembleClassifier. Got {type(model)} and method {h['method']}"
        super().__init__(h, model)
        self.model: EnsembleClassifier = model  # type hint

    # Used in EnsembleClassifier
    def forward(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        # In: (batch, nb_views, C, H, W)
        # Out: (batch, num_classes), None
        M.assert_BatchViewChannelHeightWidth_shape(self.h, x)
        distributions, _ = self.model.forward_softmaxed(x)  # (batch, num_classes)
        return distributions, None


class StdBasedSoftmaxRelaxationPredictor4Ensemble(UQPredictor):
    # Computes standard deviation between *model predictions* of the *ensemble*
    # and uses it to smooth the full prediction.
    def __init__(self, h: dict, model: EnsembleClassifier, tau_shift: Union[float, str], tau_amplifier,
                 transformation: str):
        assert h['method'] == 'ensemble' and isinstance(model, EnsembleClassifier), \
            (f"StdBasedSoftmaxRelaxationPredictor4Ensemble is only supported for EnsembleClassifier. "
             f"Got {type(model)} and method {h['method']}")
        super().__init__(h, model)
        self.model: EnsembleClassifier = model  # type hint

        if tau_shift in ['auto', 'mean'] or str(tau_shift).startswith('q'):
            tau_shift = self._compute_taw_shift(h, computation=tau_shift)
            print(f"Auto tau_shift: {tau_shift}")

        # \sigma = (tau_amplifier * standard_deviation) + tau_shift
        assert transformation in ['affine', 'exponential'], f"Unknown transformation: {transformation}"
        self.transform = transformation  # 'affine' or 'exponential'
        self.tau_shift: float = tau_shift
        self.tau_amplifier: float = tau_amplifier

        # alternative, using exponential function
        # \sigma = a(1 + r)^{x-standard_deviation} + tau_shift,
        # where a is 1 and r is tau_amplifier = growth rate (a percentage value between 0 and 1)

    def forward(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Used for uncertainty estimation; returns distributions (softmax IS applied) by smoothing full prediction based
        on standard devitation of sub-image predictions.
        In: (batch, nb_views, C, H, W)
        Out: (batch, num_classes), (batch,)
        """
        M.assert_BatchViewChannelHeightWidth_shape(self.h, x)

        yhat_logits = self.model._forward_multiple_models(x)  # (batch, num_models, num_classes)
        assert yhat_logits.dim() == 3, f"Expected 3 dimensions, got {yhat_logits.dim()}"

        standard_deviations = self._compute_standard_deviations(yhat_logits)  # (batch, 1)
        yhat_logits = self._collapse_logits(yhat_logits)  # (batch, num_classes)
        tau = self._compute_sigma_tilde(standard_deviations)  # (batch, num_classes)

        distributions = softmax(self.h, yhat_logits / tau)
        return distributions, standard_deviations.squeeze()  # (batch, num_classes), (batch,)

    def _compute_standard_deviations(self, yhat_logits: Tensor) -> Tensor:
        """
        In: (batch, num_models, num_classes)
        Out: (batch, 1)
        """
        # softplus
        # yhat_logits = nn.functional.softplus(yhat_logits)  # (batch, num_models, num_classes)
        standard_deviations = torch.std(yhat_logits, dim=1)
        return torch.mean(standard_deviations, dim=-1).unsqueeze(-1)  # (batch, 1)

    def _compute_sigma_tilde(self, standard_deviations: Tensor) -> Tensor:
        """ In: (batch, 1)
            Out: (batch, num_classes) """
        ones = torch.ones_like(standard_deviations)

        # Affine transformation
        if self.transform == 'affine':
            tau = self.tau_amplifier * (standard_deviations + self.tau_shift)
        else:  # exponential
            tau = (1 + self.tau_amplifier) ** (standard_deviations - 1) + self.tau_shift

        standard_deviations = torch.max(tau, ones)  # (batch, 1)
        return standard_deviations.expand(-1, M.num_classes(self.h))  # (batch, num_classes)

    def _collapse_logits(self, yhat_logits: Tensor) -> Tensor:
        """ In: (batch, num_models, num_classes)
            Out: (batch, num_classes) """
        return yhat_logits.mean(dim=1)  # (batch, num_classes)

    def _compute_taw_shift(self, h: dict, computation: str) -> float:
        """Iterate over the entire validation data to find a good tau_shift value."""
        from shared.data.data_loader import data_loaders

        _, val_loader, _ = data_loaders(h)
        standard_deviations = []  # (batch, 1)

        for i, (batch_clean, labels) in filter_iterator(val_loader, h['limit_val_batches'], log_progress=True):
            logits = self.model._forward_multiple_models(batch_clean)  # (batch, num_models, num_classes)
            stds = self._compute_standard_deviations(logits)  # (batch, 1)
            standard_deviations.extend(stds.squeeze().tolist())

        standard_deviations = torch.tensor(standard_deviations)
        if computation == 'auto':
            return - standard_deviations.mean().item() + 0.5  # scalar
        elif computation == 'mean':
            return - standard_deviations.mean().item()
        elif computation.startswith('q'):
            q: float = float(computation[1:]) / 100.0  # e.g. 'q25' -> 0.25
            return - torch.quantile(standard_deviations, q).item()
        else:
            raise ValueError(f"Unknown tau shift computation: {computation}")


class StdBasedSoftmaxRelaxationPredictor4EnsembleLearnedParams(
    StdBasedSoftmaxRelaxationPredictor4Ensemble):  # Standard deviation based softmax relaxation where tau_shift and tau_amplifier are learned
    def __init__(self, h: dict, model: EnsembleClassifier):
        super().__init__(h, model, 0.0, 1.0)

        self._learn_params()

    def _learn_params(self) -> Tuple[float, float]:
        self.tau_shift = nn.Parameter(torch.tensor(self.tau_shift))
        self.tau_amplifier = nn.Parameter(torch.tensor(self.tau_amplifier))

        optimizer = optim.Adam([self.tau_shift, self.tau_amplifier], lr=1e-3)
        criterion = nn.NLLLoss()

        h_temp = copy.deepcopy(self.h)
        h_temp['batch_size'] = 1024
        _, val_loader, _ = data_loaders(h_temp)
        self.model.eval()

        for epoch in range(2):  # Number of epochs can be adjusted
            for batch in val_loader:
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()

                distributions, _ = self.forward(x)  # (batch, num_classes)
                log_distributions = torch.log(distributions)  # Apply log_softmax

                loss = criterion(log_distributions, y)
                loss.backward()
                optimizer.step()

            print(self.tau_amplifier, self.tau_shift, self.tau_amplifier.grad, self.tau_shift.grad)
            print(f"Epoch: {epoch}, Loss: {loss.item()}")

        print(f"Learned tau_shift: {self.tau_shift.item()}, tau_amplifier: {self.tau_amplifier.item()}")

        return self.tau_shift.item(), self.tau_amplifier.item()


if __name__ == '__main__':
    # set seed
    # torch.manual_seed(0)
    #
    # # --dataset RADIO --use_wandb False --limit_train_batches 0.1 --limit_val_batches 0.1 --limit_test_batches 1.0 --num_views 1 --fast_dev_run True --redundancy_method identity --redundancy_method_params {} --remove_first_n_samples 200 --epochs 10 --train True --latent_dims "[512, 512, 512, 512, 512]" --latent_strides "[5, 4, 1, 1, 1]" --latent_kernel_sizes "[10, 8, 4, 4, 4]" --latent_padding [2,2,2,2,1] --dataset_path ./../Smooth-InfoMax\datasets --batch_size 8
    # _h = {
    #     'dataset': 'RADIO',
    #     'num_views': 13,
    #     'latent_dims': [512, 512, 512, 512, 512],
    #     'latent_strides': [5, 4, 1, 1, 1],
    #     'latent_kernel_sizes': [10, 8, 4, 4, 4],
    #     'latent_padding': [2, 2, 2, 2, 1],
    #     'alpha': 0,
    #     'redundancy_method_uncertainty_kernel_size': 20,
    # }
    # c = MultiInputClassifier.create_instance(_h)
    # # print(c)
    #
    # # 1880 = 2080 - 200
    # # if 1880 -> l' is 94
    # batch = torch.randn(32, _h['num_views'], 2, 1880)  # (batch, nb_views, C, L)
    # out, latent = c.forward(batch)
    # print(out.shape)
    #
    # std_based_predictor: StdBasedSoftmaxRelaxationPredictor = StdBasedSoftmaxRelaxationPredictor(_h, c, tau_shift=-4.5)
    #
    # out, std = std_based_predictor.forward(batch)
    #
    # print(out.shape)

    logits = torch.rand(8, 81, 20)  # (batch, seq_len, num_classes)
    logits_avg_pooled = nn.functional.avg_pool1d(logits.permute(0, 2, 1), kernel_size=40, stride=1).permute(0, 2, 1)
    print(logits_avg_pooled.shape)
