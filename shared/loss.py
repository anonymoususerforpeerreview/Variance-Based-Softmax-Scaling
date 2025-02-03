import lightning as L
import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn
from shared.data.dataset_meta import DatasetMeta as M
from shared.utils import Sanity
from abc import ABC, abstractmethod


class LossFactory:
    @staticmethod
    def create_loss(h: dict) -> 'Loss':
        assert h['loss'] in ['cross_entropy', 'softplus_based_cross_entropy'], f"Loss {h['loss']} not supported."
        if h['loss'] == 'cross_entropy':
            return CrossEntropyOptionalRegularizer(h['alpha'])
        elif h['loss'] == 'softplus_based_cross_entropy':
            return SoftPlusBasedCrossEntropy()
        else:
            raise ValueError(f"Loss {h['loss']} not supported.")


class Loss(L.LightningModule, ABC):
    @abstractmethod
    def forward(self, y_hat: Tensor, y: Tensor, latent_representations: list, nb_views: int):
        pass


class CrossEntropyOptionalRegularizer(Loss):
    def __init__(self, alpha: float):
        super(CrossEntropyOptionalRegularizer, self).__init__()
        assert 0.0 <= alpha <= 1.0, f"Alpha must be in [0, 1], got {alpha}."

        self.cross_entropy = nn.CrossEntropyLoss()
        self.alpha = alpha

    def forward(self, y_hat: Tensor, y: Tensor, latent_representations: list, nb_views: int):
        """
        In: y_hat (batch, nb_views, num_classes), y (batch) where y_hat are the logits
        Out: Scalar
        """
        y = y.unsqueeze(1).expand(-1, nb_views)  # (batch, nb_views)

        # reshape to (batch * nb_views, num_classes)
        y_hat = y_hat.view(-1, y_hat.size(2))
        y = y.reshape(-1)

        if self.alpha == 1.0:
            return self.cross_entropy(y_hat, y)
        else:
            return (
                    self.alpha * self.cross_entropy(y_hat, y) +
                    (1 - self.alpha) * self.distinct_path_regularization(latent_representations))

        # return (self.cross_entropy(y_hat, y) +
        #         self.distinct_path_regularization(latent_representations))

    # L2 distance between all pairs of views
    # def distinct_path_regularization(self, latent_representations):
    #     """Compute the regularization term to penalize similar representations.
    #     In: List of latent representations of shape (batch, nb_views, C, H, W)
    #     Out: Scalar
    #     """
    #     reg_loss = 0.0
    #     for l in latent_representations:
    #         assert l.dim() == 5, f"Latent representation shape is {l.shape}, expected 5 dimensions."
    #         # Compute pairwise distances between views
    #         batch, nb_views, *dims = l.shape
    #         l = l.view(batch, nb_views, -1)  # Flatten spatial dimensions
    #
    #         distances = torch.cdist(l, l, p=2)  # Pairwise L2 distances
    #         reg_loss += distances.mean()
    #     return reg_loss

    def distinct_path_regularization(self, latent_representations):
        """Compute the regularization term to penalize similar representations.
        In: List of latent representations of shape (batch, nb_views, C, H, W)
        Out: Scalar
        """
        reg_loss = 0.0
        for l in latent_representations:
            M.assert_BatchViewChannelHeightWidth_shape(self.h, l)
            # assert l.dim() == 5, f"Latent representation shape is {l.shape}, expected 5 dimensions."
            # Compute pairwise dot products between views
            batch, nb_views, *dims = l.shape
            l = l.view(batch, nb_views, -1)  # Flatten spatial dimensions

            # Normalize the vectors
            l = F.normalize(l, p=2, dim=-1)

            # Compute pairwise dot products
            dot_products = torch.bmm(l, l.transpose(1, 2))  # (batch, nb_views, nb_views)

            # We want to penalize high similarity, so we take the mean of the dot products
            reg_loss += dot_products.mean()
        return reg_loss


def softplus_based_softmax(logits: Tensor) -> Tensor:
    """
    Apply the softmax function to the logits.
    In: logits (batch, nb_classes)
    Out: probabilities (batch, nb_classes)
    """
    return F.softplus(logits) / F.softplus(logits).sum(dim=1, keepdim=True)


class SoftPlusBasedCrossEntropy(Loss):
    def __init__(self):
        super(SoftPlusBasedCrossEntropy, self).__init__()

    def forward(self, y_hat: Tensor, y: Tensor, _: list, nb_views: int):
        """
        In: y_hat (batch, nb_views, num_classes), y (batch) where y_hat are the logits
        Out: Scalar
        """
        y = y.unsqueeze(1).expand(-1, nb_views)  # (batch, nb_views)

        # reshape to (batch * nb_views, num_classes)
        y_hat = y_hat.view(-1, y_hat.size(2)) # (batch * nb_views, num_classes)
        y = y.reshape(-1) # (batch * nb_views)

        # softplus based softmax
        # y_hat = F.softplus(y_hat) / F.softplus(y_hat).sum(dim=1, keepdim=True)  # (batch * nb_views, num_classes)
        y_hat = softplus_based_softmax(y_hat)

        Sanity.assert_softmax_is_applied(y_hat)  # verifies that the sum of the probabilities is 1

        # TODO cannot just call self.cross_entropy(y_hat, y) because of the softplus
        # Compute the negative log likelihood of the normalized distribution
        log_probs = torch.log(y_hat + 1e-12)  # Add a small value for numerical stability
        loss = -log_probs[torch.arange(y.size(0)), y].mean()  # Gather the correct class probabilities and take the mean
        return loss


if __name__ == "__main__":
    torch.manual_seed(0)
    # loss = CrossEntropyOptionalRegularizer(1)
    # b = 1
    # v = 5
    # y_hat = torch.randn(b, v, 10)  # (batch, nb_views, num_classes)
    # y = torch.randint(0, 10, (b, ))  # (batch, nb_views)
    # latent_representations = [torch.randn(b, v, 64, 7, 7), torch.randn(b, v, 128, 3, 3)]
    # loss(y_hat, y, latent_representations, v)

    loss = SoftPlusBasedCrossEntropy()
    b = 1
    v = 5
    y_hat = torch.randn(b, v, 10)  # (batch, nb_views, num_classes)
    y = torch.randint(low=0, high=10, size=(b,))  # (batch, nb_views)
    loss(y_hat, y, None, v)
