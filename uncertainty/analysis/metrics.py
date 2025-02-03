from torch.utils.data import DataLoader

from shared.data import filter_iterator
from shared.data.noise import NoiseFactory
from shared.utils import Sanity
from uncertainty.analysis.uq import UQ
from uncertainty.uq_through_redundancy.uq_predictor import UQPredictor
import torch
import torch.nn.functional as F


class Metrics:
    @staticmethod
    def compute_accuracy(uq_predictor: UQPredictor, test_loader: DataLoader, limit_batches: float,
                         noise_type: str = None, noise_level: float = 0.0) -> float:
        """
        Compute the accuracy of the model on the test set.
        """
        correct = 0
        total = 0
        for i, (images, labels) in filter_iterator(test_loader, limit_batches):
            if noise_type is not None:
                images = NoiseFactory.apply_noise(images, noise_type, noise_level)
            distributions, _ = uq_predictor.forward(images)  # (batch, num_classes)
            Sanity.assert_softmax_is_applied(distributions)

            _, predicted = torch.max(distributions, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        return correct / total

    @staticmethod
    def compute_nll(uq_predictor: UQPredictor, test_loader: DataLoader, limit_batches: float) -> float:
        """
        Compute the negative log likelihood of the model on the test set.
        Lower is better.
        """
        nll_loss = torch.nn.NLLLoss()
        total_loss = 0.0
        total_samples = 0

        for i, (images, labels) in filter_iterator(test_loader, limit_batches):
            distributions, _ = uq_predictor.forward(images)  # (batch, num_classes)
            Sanity.assert_softmax_is_applied(distributions)

            loss = nll_loss(distributions, labels)
            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)

        return total_loss / total_samples

    @staticmethod
    def compute_brier_score(uq_predictor: UQPredictor, test_loader: DataLoader, limit_batches: float,
                            noise_type: str = None, noise_level: float = 0.0) -> float:
        """
        Compute the Brier score of the model on the test set.
        The Brier score evaluates the entire probability distribution,
        not just the max probability. Lower values indicate better-calibrated predictions.

        """
        total_brier_score = 0.0
        total_samples = 0

        for i, (images, labels) in filter_iterator(test_loader, limit_batches):
            if noise_type is not None:
                images = NoiseFactory.apply_noise(images, noise_type, noise_level)
            distributions, _ = uq_predictor.forward(images)  # (batch, num_classes)
            Sanity.assert_softmax_is_applied(distributions)

            one_hot_labels = F.one_hot(labels, num_classes=distributions.size(1)).float()
            brier_score = torch.sum((distributions - one_hot_labels) ** 2, dim=1).mean()
            total_brier_score += brier_score.item() * labels.size(0)
            total_samples += labels.size(0)

        return total_brier_score / total_samples

    @staticmethod
    def compute_cross_entropy(uq_predictor: UQPredictor, test_loader: DataLoader, limit_batches: float) -> float:
        """
        Compute the cross entropy of the model on the test set.
        Lower is better.
        """
        cross_entropy_loss = torch.nn.CrossEntropyLoss()
        total_loss = 0.0
        total_samples = 0

        for i, (images, labels) in filter_iterator(test_loader, limit_batches):
            distributions, _ = uq_predictor.forward(images)  # (batch, num_classes)
            Sanity.assert_softmax_is_applied(distributions)

            # apply log to distributions as CrossEntropyLoss expects log probabilities
            loss = cross_entropy_loss(torch.log(distributions), labels)
            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)

        return total_loss / total_samples

    @staticmethod
    def compute_ece(uq_predictor: UQPredictor, test_loader: DataLoader, limit_batches: float,
                    n_bins: int = 15,
                    noise_type: str = None,
                    noise_level: float = 0.0
                    ) -> float:
        """
        Compute the Expected Calibration Error (ECE) of the model on the test set.
        """
        total_ece = 0.0
        total_samples = 0

        for i, (images, labels) in filter_iterator(test_loader, limit_batches):
            if noise_type is not None:
                images = NoiseFactory.apply_noise(images, noise_type, noise_level)

            distributions, _ = uq_predictor.forward(images)  # (batch, num_classes)
            Sanity.assert_softmax_is_applied(distributions)

            confidences, predictions = torch.max(distributions, 1)  # (batch)
            accuracies = predictions.eq(labels)  # (batch)

            bin_boundaries = torch.linspace(0, 1, n_bins + 1)
            ece = 0.0

            for bin_lower, bin_upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
                in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
                prop_in_bin = in_bin.float().mean().item()
                if prop_in_bin > 0:
                    accuracy_in_bin = accuracies[in_bin].float().mean().item()
                    avg_confidence_in_bin = confidences[in_bin].mean().item()
                    ece += abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

            total_ece += ece * labels.size(0)
            total_samples += labels.size(0)

        return total_ece / total_samples

    @staticmethod
    def compute_entropy(uq_predictor: UQPredictor, test_loader: DataLoader, limit_batches: float,
                        noise_type: str = None, noise_level: float = 0.0
                        ) -> float:
        """
        Compute the entropy of the model on the test set.
        """
        total_entropy = 0.0
        total_samples = 0

        for i, (images, labels) in filter_iterator(test_loader, limit_batches):
            if noise_type is not None:
                images = NoiseFactory.apply_noise(images, noise_type, noise_level)

            distributions, _ = uq_predictor.forward(images)  # (batch, num_classes)
            Sanity.assert_softmax_is_applied(distributions)

            entropy = UQ.compute_entropy(distributions)  # (batch)
            total_entropy += entropy.sum().item()
            total_samples += labels.size(0)  # += (batch)

        return total_entropy / total_samples

    @staticmethod
    def compute_kl_divergence_vs_uniform(uq_predictor: UQPredictor, test_loader: DataLoader, limit_batches: float,
                                         noise_type: str = None, noise_level: float = 0.0
                                         ) -> float:
        """
        Compute the KL divergence of the model on the test set.
        """
        total_kl_divergence = 0.0
        total_samples = 0

        for i, (images, labels) in filter_iterator(test_loader, limit_batches):
            if noise_type is not None:
                images = NoiseFactory.apply_noise(images, noise_type, noise_level)

            distributions, _ = uq_predictor.forward(images)
            Sanity.assert_softmax_is_applied(distributions)

            kl_divergence = UQ.compute_kl_divergence_vs_uniform(distributions)  # (batch)
            total_kl_divergence += kl_divergence.sum().item()
            total_samples += labels.size(0)  # += (batch)

        return total_kl_divergence / total_samples
