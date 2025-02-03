import torch
import numpy as np
import torch.nn.functional as F
from numpy import ndarray
from torch import Tensor
from torch.utils.data import DataLoader
from typing import Tuple, Optional, Union, List

from shared.data import filter_iterator
from shared.data.noise import NoiseFactory
from shared.hyperparameters import Hyperparameters
from shared.utils import Sanity
from uncertainty.analysis.visual import Visual
from uncertainty.uq_through_redundancy.multi_input_classifier import MultiInputClassifier
from shared.data.dataset_meta import DatasetMeta as M
from shared.chunk_writer import ChunkWriter as CW
from uncertainty.uq_through_redundancy.uq_predictor import UQPredictor


class UQ:  # Uncertainty Quantification
    @staticmethod
    # Function to compute model's uncertainty (e.g., softmax variance)
    def compute_entropy(probs):
        """
        Returns entropy(probabilities_averaged_across_views).
        in: probs: (batch_size, num_classes)
        out: uncertainty: (batch_size)
        """
        assert probs.dim() == 2, f"Expected 2 dimensions, got {probs.dim()}."

        Sanity.assert_softmax_is_applied(probs)

        # Compute entropy across the class probabilities
        epsilon = 1e-10  # Small value to avoid log(0)
        uncertainty = -(probs * torch.log(probs + epsilon)).sum(dim=1)  # (batch_size)

        return uncertainty  # (batch_size)

    @staticmethod
    def compute_kl_divergence(probs1, probs2):
        """
        Returns KL divergence between probs1 and probs2.
        in: probs1, probs2: (batch_size, num_classes)
        out: kl_divergence: (batch_size)
        """
        assert probs1.dim() == 2, f"Expected 2 dimensions, got {probs1.dim()}."
        assert probs2.dim() == 2, f"Expected 2 dimensions, got {probs2.dim()}."

        # assert softmax is already applied
        assert torch.allclose(probs1.sum(dim=1),
                              torch.ones(probs1.shape[0], device=probs1.device)), \
            f"Expected the sum of the probabilities to be 1, got {probs1.sum(dim=1)}."
        assert torch.allclose(probs2.sum(dim=1),
                              torch.ones(probs2.shape[0], device=probs2.device)), \
            f"Expected the sum of the probabilities to be 1, got {probs2.sum(dim=1)}."

        # Compute KL divergence across the class probabilities
        epsilon = 1e-10  # Small value to avoid log(0)
        kl_divergence = (probs1 * (torch.log(probs1 + epsilon) - torch.log(probs2 + epsilon))).sum(dim=1)

        return kl_divergence  # (batch_size)

    @staticmethod
    def compute_kl_divergence_vs_uniform(probs):
        """
        Returns KL divergence between probs and uniform distribution.
        in: probs: (batch_size, num_classes)
        out: kl_divergence: (batch_size)
        """
        assert probs.dim() == 2, f"Expected 2 dimensions, got {probs.dim()}."

        # assert softmax is already applied
        assert torch.allclose(probs.sum(dim=1),
                              torch.ones(probs.shape[0], device=probs.device)), \
            f"Expected the sum of the probabilities to be 1, got {probs.sum(dim=1)}."

        # Compute KL divergence across the class probabilities
        num_classes = probs.size(1)
        uniform_probs = torch.ones_like(probs) / num_classes
        epsilon = 1e-10  # Small value to avoid log(0)
        kl_divergence = (probs * (torch.log(probs + epsilon) - torch.log(uniform_probs + epsilon))).sum(dim=1)

        return kl_divergence  # (batch_size)

    @staticmethod
    def _final_prediction_from_multiple_view_logits(logits: Tensor, apply_softmax: bool) -> Tuple[Tensor, Tensor]:
        """
        In: outputs: (batch_size, num_classes)
        Out: final_predictions: (batch_size), average_variance: (batch_size)

        """
        assert logits.dim() == 2, f"Expected 2 dimensions, got {logits.dim()}."

        # softmax the predictions
        if apply_softmax:
            predictions = F.softmax(logits, dim=1)  # (batch_size, num_classes)
        else:
            predictions = logits

        # 2) compute the variance of the predictions # NO LONGER SUPPORTED
        average_variance = torch.zeros(predictions.size(0), device=predictions.device)

        # 3) get predictions argmax
        # (batch_size, num_classes) -> (batch_size)
        final_predictions = (predictions.argmax(dim=1))

        return final_predictions, average_variance

    @staticmethod
    def _variances_vs_accuracy_per_input_img(classifier: MultiInputClassifier, batch: Tuple[Tensor, Tensor]) -> Tensor:
        """Returns tensor of shape (batch, 2),
        where the first column is the variance and the second column
        is 1 or 0 if the prediction is correct or not."""

        x, y = batch
        M.assert_BatchViewChannelHeightWidth_shape(classifier.h, x)

        # 1) per view, compute the predictions
        probs, std = classifier.forward_multiple_predictive_uncertainty(x)  # 2x (batch_size, num_views, num_classes)
        avg_std = std.mean(dim=(1, 2))  # (batch_size)

        final_predictions, _ = UQ._final_prediction_from_multiple_view_logits(probs, apply_softmax=False)

        # 4) compute the accuracy of the mode with the labels
        accuracy = (final_predictions == y)  # (batch_size)

        # 5) stack the variance and accuracy
        stack = torch.stack((avg_std, accuracy), dim=1)  # (batch_size, 2)
        return stack

    @staticmethod
    def variances_vs_accuracy_per_input_img(classifier: MultiInputClassifier, data_loader: DataLoader,
                                            limit_batches: float, chunk_size=10) -> Tensor:
        """Returns tensor of shape (nb_files_in_dataset, 2),
        where the first column is the variance and the second column
        is 1 or 0 if the prediction is correct or not."""

        assert 0 <= limit_batches <= 1, f"limit_batches must be between 0 and 1, got {limit_batches}."
        cw = CW()  # ChunkWriter for saving the data in chunks (to avoid memory issues)

        classifier.eval()
        var_vs_accuracy = []

        for i, batch in filter_iterator(data_loader, limit_batches, log_progress=True):
            batch_var_vs_accuracy = UQ._variances_vs_accuracy_per_input_img(classifier, batch)
            var_vs_accuracy.append(batch_var_vs_accuracy)

            # Save the data in chunks to avoid memory issues using the ChunkWriter
            if len(var_vs_accuracy) >= chunk_size:
                cw.save_chunks(var_vs_accuracy, chunk_size, "var_vs_accuracy")
                var_vs_accuracy = []

        # Save the remaining data
        if var_vs_accuracy:
            cw.save_chunks(var_vs_accuracy, chunk_size, "var_vs_accuracy_final")

        return cw.load_and_delete_chunks("var_vs_accuracy")

    @staticmethod
    def apply_f_to_uq_score_all_noise_levels(model: UQPredictor, test_loader, limit_batches: float, function: callable,
                                             noise_type: str,
                                             is_1d_signal: bool,
                                             min_noise: float = 0, max_noise: float = 1.0,
                                             display_samples: bool = False) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Iterates over different noise levels and computes the mean entropy and accuracy of the model's predictions.
        :param model:
        :param test_loader:
        :param limit_batches:
        :param display_samples:
        :return:
        """
        assert 0 <= limit_batches <= 1, f"limit_batches must be between 0 and 1, got {limit_batches}."

        model.eval()

        noise_levels, mean_uncertainties, mean_accuracies, mean_std, imgs_per_noise_level = \
            UQ._retrieve_generic_f_data_all_noise_levels(model, test_loader, function, noise_type, limit_batches,
                                                         is_1d_signal,
                                                         min_noise,
                                                         max_noise)

        if display_samples:
            if is_1d_signal:
                Visual.plot_signal(imgs_per_noise_level,
                                   title=f"Noise Levels: {[round(noise, 2) for noise in noise_levels]}")
            else:  # 2D images, imgs_per_noise_level: (C, H, W'), e.g, (3, 32, 32*8) for 8 images
                Visual.plot_img(imgs_per_noise_level,
                                title=f"Noise Levels: {[round(noise, 2) for noise in noise_levels]}")

        return noise_levels, mean_uncertainties, mean_accuracies, mean_std, imgs_per_noise_level
        # return noise_levels, np.array(mean_uncertainties), np.array(mean_accuracies), imgs_per_noise_level

    @staticmethod
    def _retrieve_generic_f_data_all_noise_levels(model: UQPredictor, test_loader, function, noise_type: str,
                                                  limit_batches: float,
                                                  is_1d_signal: bool,
                                                  min_noise: float, max_noise: float) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        noise_levels: np.ndarray = np.linspace(min_noise, max_noise, 20)
        mean_uncertainties = []
        mean_accuracies = []
        mean_stds = []
        imgs_per_noise_level = None

        for noise_factor in noise_levels:
            print(f"Computing entropy for noise level: {noise_factor}/{max_noise}. Tot: {len(noise_levels)}")

            mean_uncertainty, mean_accuracy, mean_std, imgs = \
                UQ._retrieve_generic_f_data(model, test_loader, function, noise_type, limit_batches, is_1d_signal,
                                            noise_factor)

            mean_uncertainties.append(mean_uncertainty)
            mean_accuracies.append(mean_accuracy)
            mean_stds.append(mean_std)
            if imgs_per_noise_level is None:
                imgs_per_noise_level = imgs
            else:
                if is_1d_signal:
                    imgs_per_noise_level = torch.cat((imgs_per_noise_level, imgs), dim=1)
                else:
                    imgs_per_noise_level = torch.cat((imgs_per_noise_level, imgs), dim=2)

        mean_uncertainties = np.array(mean_uncertainties).reshape(-1)
        mean_accuracies = np.array(mean_accuracies).reshape(-1)
        mean_stds = np.array(mean_stds).reshape(-1)
        return noise_levels, mean_uncertainties, mean_accuracies, mean_stds, imgs_per_noise_level

    @staticmethod
    def _retrieve_generic_f_data(model: UQPredictor, test_loader, function: callable, noise_type: str,
                                 limit_batches: float, is_1d_signal: bool, noise_factor: float) -> \
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the mean uncertainty and accuracy of the model's predictions.
        Out: mean_uncertainty: scalar, mean_accuracy: scalar, imgs_per_noise_level: Tensor

        Note: mean_std is only used for StdBasedSoftmaxRelaxationPredictor. The other predictors do not return the std.
        Std is useful to see if there is a correlation between noisy inputs and the std of patch-based predictions.
        For other predictors, mean_std is always 0.
        """
        all_uncertainties = torch.tensor([], dtype=torch.float32, device=model.device)
        all_accuracies = torch.tensor([], dtype=torch.float32, device=model.device)
        all_standard_deviations = torch.tensor([], dtype=torch.float32, device=model.device)
        imgs_per_noise_level = None

        for i, (images, labels) in filter_iterator(test_loader, limit_batches):
            # images: (batch_size, num_views, C, L) or (batch_size, num_views, C, H, W)
            noisy_images = NoiseFactory.apply_noise(images, noise_type, noise_factor)
            # (batch_size, num_views, C, L) or (batch_size, num_views, C, H, W)
            if i == 0:  # Display the first batch of images
                imgs_per_noise_level = UQ._plot_first_image(noisy_images, imgs_per_noise_level, is_1d_signal)

            with torch.no_grad():
                probs, std = model.forward(noisy_images)  # (b_size, num_classes), (b_size)

                # entropies = UQ.compute_entropy(probs)
                scores = function(probs)  # e.g. entropies
                all_uncertainties = torch.cat((all_uncertainties, scores))
                if std is not None:
                    all_standard_deviations = torch.cat((all_standard_deviations, std))

                predictions, _ = UQ._final_prediction_from_multiple_view_logits(probs, apply_softmax=False)
                correct_predictions = (predictions == labels).float()
                all_accuracies = torch.cat((all_accuracies, correct_predictions))

        mean_uncertainty = torch.tensor([all_uncertainties.mean().item()])
        mean_accuracy = torch.tensor([all_accuracies.mean().item()])
        if all_standard_deviations.nelement() > 0:
            mean_std = torch.tensor([all_standard_deviations.mean().item()])
        else:
            mean_std = torch.tensor([0.0])

        return mean_uncertainty, mean_accuracy, mean_std, imgs_per_noise_level

    @staticmethod
    def _plot_first_image(noisy_images, imgs_per_noise_level, is_1d_signal):
        # imgs_per_noise_level: (batch_size, num_views, C, L) or (batch_size, num_views, C, H, W)
        if imgs_per_noise_level is None:  # First iteration
            # Initialize for 2D images
            imgs_per_noise_level = noisy_images[0][0]  # ([0][0]: batch, view)
        else:  # Concatenate the rest of the images
            if is_1d_signal:
                # Concatenate for 1D signals
                imgs_per_noise_level = torch.cat((imgs_per_noise_level, noisy_images[0][0]), dim=1)
                # Add a 0 padding to separate the signals
                padding = torch.zeros_like(noisy_images[0][0])
                imgs_per_noise_level = torch.cat((imgs_per_noise_level, padding, noisy_images[0][0]), dim=1)
            else:
                # Concatenate for 2D images
                imgs_per_noise_level = torch.cat((imgs_per_noise_level, noisy_images[0][0]), dim=2)
        return imgs_per_noise_level

    @staticmethod
    def uncertainty_per_avg_accuracy_line(uq_predictor: UQPredictor, test_loader, num_bins: int,
                                          limit_batches: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes the uncertainty per average accuracy line.
        :param classifier:
        :param test_loader:
        :param num_bins:
        :param limit_batches:
        :return:
        """
        assert 0 <= limit_batches <= 1, f"limit_batches must be between 0 and 1, got {limit_batches}."

        uq_predictor.eval()

        # Initialize bins and accuracy counters
        uncertainty_bins = np.linspace(0, 1, num_bins + 1)
        accuracy_counts = np.zeros(num_bins)
        total_counts = np.zeros(num_bins)

        for i, (images, labels) in filter_iterator(test_loader, limit_batches):
            with torch.no_grad():
                probs, _ = uq_predictor.forward(images)  # (batch_size, num_classes)
                max_probs, predictions = probs.max(dim=1)  # (batch_size)

                correct_predictions = (predictions == labels).float()  # (batch_size) e.g., [1, 0, 1, 1, 0, ...]

                for prob, correct in zip(max_probs, correct_predictions):
                    # prob: scalar, correct: scalar
                    bin_index = np.digitize(prob.item(), uncertainty_bins) - 1  # -1 to account for 0-based indexing
                    # TODO: Check if this is correct (IT WAS GENERATED BY THE COPILOT)
                    bin_index = min(bin_index, num_bins - 1)  # Clamp to the maximum valid index
                    accuracy_counts[bin_index] += correct.item()
                    total_counts[bin_index] += 1

        avg_accuracies = np.divide(accuracy_counts, total_counts, out=np.zeros_like(accuracy_counts),
                                   where=total_counts != 0)

        return uncertainty_bins[:-1], avg_accuracies, total_counts

    @staticmethod
    @torch.no_grad()
    def compute_accuracy_kernel_based_inputs(noise_level, noise_type, kernel_size, uq_predictor: UQPredictor,
                                             test_loader: DataLoader, limit_batches: float) -> float:
        all_accuracies = torch.tensor([], dtype=torch.float32, device=uq_predictor.device)
        for i, (images, labels) in filter_iterator(test_loader, limit_batches):
            noisy_images = NoiseFactory.apply_noise(images, noise_type,
                                                    noise_level)  # (batch_size, num_views, C, L) or (batch_size, num_views, C, H, W)
            # (bsize*num_views, ... )
            b_size = noisy_images.size(0)
            nb_views = noisy_images.size(1)
            noisy_images = noisy_images.view(b_size * nb_views, *noisy_images.size()[2:])

            logits, _ = uq_predictor.model._forward_single_with_moving_average(noisy_images,
                                                                               kernel_size)  # (batch_size*num_views, L', num_classes)

            # image data
            if M.is_1d_signal(uq_predictor.h):
                assert logits.dim() == 3, f"Expected 3 dimensions, got {logits.dim()}."  # (batch_size*num_views, num_classes, L')
            else:
                assert logits.dim() == 4, f"Expected 4 dimensions, got {logits.dim()}."  # (batch_size*num_views, num_classes, H', W')

            logits = logits.reshape(-1, logits.size(-1))  # (batch_size*num_views*H'*W', num_classes)

            # 1) compute the predictions
            predictions, _ = UQ._final_prediction_from_multiple_view_logits(
                logits, apply_softmax=True)  # (batch_size*num_views*H'*W') or (batch_size*num_views*L')

            # 2) compute the accuracy of the mode with the labels
            # labels: (batch_size)
            labels = labels.repeat(nb_views)  # (batch_size*num_views)

            # H*W or L
            L = int(predictions.size(0) / (b_size * nb_views))
            labels = labels.repeat_interleave(L)  # (batch_size*num_views*H'*W') or (batch_size*num_views*L')
            accuracy = (predictions == labels).float()  # (batch_size*num_views*H'*W')

            all_accuracies = torch.cat((all_accuracies, accuracy))

        return all_accuracies.mean().item()

    @staticmethod
    @torch.no_grad()
    def compute_logits_kernel_based_inputs(noise_level, noise_type, kernel_size, uq_predictor: UQPredictor,
                                           test_loader: DataLoader, limit_batches: float, n_elements: int):
        """
        In: noise_level, noise_type, kernel_size, uq_predictor, test_loader, limit_batches, n_elements
        Out:
            logits: (n_elements, num_views, L', num_classes) or (n_elements, num_views, H'*W', num_classes),
            labels: (n_elements)
        """
        all_logits = []
        all_labels = []
        total_batches = len(test_loader)
        step = max(1, total_batches // n_elements)
        if limit_batches < 1:
            print("Warning: compute_logits_kernel_based_inputs(): limit_batches is less than 1. "
                  "The number of elements might be less than n_elements.")

        for i, (images, labels) in filter_iterator(test_loader, limit_batches):
            if i % step == 0 and len(all_logits) < n_elements:
                noisy_images = NoiseFactory.apply_noise(images, noise_type, noise_level)

                # (bsize, num_views, ... )
                b_size = noisy_images.size(0)
                nb_views = noisy_images.size(1)
                noisy_images = noisy_images.view(b_size * nb_views, *noisy_images.size()[2:])

                logits, _ = uq_predictor.model._forward_single_with_moving_average(noisy_images, kernel_size)

                if M.is_1d_signal(uq_predictor.h):
                    assert logits.dim() == 3, f"Expected 3 dimensions, got {logits.dim()}."
                else:
                    assert logits.dim() == 4, f"Expected 4 dimensions, got {logits.dim()}."

                logits = logits.reshape(-1, logits.size(-1))
                num_classes = logits.size(-1)
                spatial_dim = logits.size(0) // b_size

                logits = logits.view(b_size, nb_views, spatial_dim, num_classes).cpu().numpy()
                all_logits.append(logits[0])  # append the first element
                all_labels.append(labels[0].cpu().numpy())

        return np.array(all_logits), np.array(all_labels)

    @staticmethod
    @torch.no_grad()
    def compute_predicted_distributions(noise_level: float, noise_type: str, uq_predictor: UQPredictor, test_loader,
                                        limit_batches: float, n_elements: int) -> Tuple[ndarray, ndarray]:
        """
        In: noise_level, noise_type, uq_predictor, test_loader, limit_batches, n_elements
        Out: predicted_distributions: (n_elements, num_classes), labels: (n_elements)
        """
        all_distributions = []
        all_labels = []
        total_batches = len(test_loader)
        step = max(1, total_batches // n_elements)
        if limit_batches < 1:
            print("Warning: compute_predicted_distributions(): limit_batches is less than 1. "
                  "The number of elements might be less than n_elements.")

        for i, (images, labels) in filter_iterator(test_loader, limit_batches):
            if i % step == 0 and len(all_distributions) < n_elements:
                noisy_images = NoiseFactory.apply_noise(images, noise_type, noise_level)

                # (bsize, num_views, ... )
                b_size = noisy_images.size(0)
                nb_views = noisy_images.size(1)

                probs, _ = uq_predictor.forward(noisy_images)  # (b_size, num_classes)
                Sanity.assert_softmax_is_applied(probs)

                # probs = probs.reshape(-1, probs.size(-1)) # (batch_size*num_views*H'*W', num_classes)
                # num_classes = probs.size(-1)
                # spatial_dim = probs.size(0) // b_size

                # probs = probs.view(b_size, nb_views, spatial_dim, num_classes).cpu().numpy()
                probs = probs.cpu().numpy()
                all_distributions.append(probs[0])
                all_labels.append(labels[0].cpu().numpy())
                # all_distributions.append(probs)
                # all_labels.append(labels.cpu().numpy())

        return np.array(all_distributions), np.array(all_labels)

    # @staticmethod
    # @torch.no_grad()
    # def compute_ece_under_distributional_shift(noise_level, noise_type, uq_predictor, test_loader,
    #                                            limit_batches):
    #     """
    #     Out: ece: scalar
    #     """
    #     return Metrics.compute_ece(uq_predictor, test_loader, limit_batches)

    @staticmethod
    @torch.no_grad()
    def apply_f_to_all_noise_levels(function: callable, start: float = 0, stop: float = 1, num: int = 20) \
            -> Union[Tuple[ndarray, ndarray], Tuple[ndarray, ndarray, ndarray]]:
        # noise_levels = np.linspace(0, 0.22, 20) # used these values for only visualizing distribution flexibility.
        # noise_levels = np.linspace(0, 0.7, 60)
        noise_levels = np.linspace(start, stop, num)
        results1 = []  # accumulate the results for each noise level
        results2 = []  # accumulate the results for each noise level
        for noise_level in noise_levels:
            print(f"Noise level: {noise_level:.2f}")
            result = function(noise_level)
            if isinstance(result, tuple):  # compute_logits_kernel_based_inputs
                results1.append(result[0])
                results2.append(result[1])
            else:  # compute_accuracy_kernel_based_inputs
                results1.append(result)

        if len(results2) == 0:  #
            return noise_levels, np.array(results1)
        else:  # in case of tuple, called in `compute_logits_kernel_based_inputs`
            return noise_levels, np.array(results1), np.array(results2)


if __name__ == '__main__':
    import torch

    torch.manual_seed(0)
    kernel_size = 20

    h = {
        'dataset': 'RADIO',
        'dataset_path': './../Smooth-InfoMax\datasets',
        'batch_size': 32,
        'num_views': 1,
        'num_workers': 1,
        'remove_first_n_samples': 200,

        'latent_dims': [512, 512, 512, 512, 512],
        'latent_strides': [5, 4, 1, 1, 1],
        'latent_kernel_sizes': [10, 8, 4, 4, 4],
        'latent_padding': [2, 2, 2, 2, 1],
        'alpha': 0,
        'redundancy_method': 'identity',
        'redundancy_method_params': {},
        'redundancy_method_uncertainty_kernel_size': kernel_size,
        'log_path': './logs/',
        'checkpoint_path': './logs/saved_models',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }
    c = MultiInputClassifier.create_instance(h).to(h['device'])
    c.load()
    # batch = torch.randn(h['batch_size'], h['num_views'], 2, 1880)  # (batch, nb_views, C, L)

    # get batch from real dataset instead
    from shared.data.data_loader import data_loaders

    _, _, test_loader = data_loaders(h)
    for i, (batch_orig, labels) in filter_iterator(test_loader, 1.0, log_progress=True):
        break

    for noise_level in np.linspace(0, 1, 20):
        print(f"Noise level: {noise_level:.2f}")

        batch = NoiseFactory.apply_noise(batch_orig, "gaussian", noise_level)

        with torch.no_grad():
            probs, _ = c.forward_multiple_predictive_uncertainty(batch)  # (b_size, views, num_classes)
            # scores = UQ.compute_entropy(probs)  # (batch_size)

            # predictions, _ = UQ._final_prediction_from_multiple_view_logits(probs, apply_softmax=False)
