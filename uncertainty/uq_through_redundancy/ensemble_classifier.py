from copy import deepcopy
from typing import Optional, List
import os
import torch
from shared.abstract_model import ANN
from uncertainty.uq_through_redundancy.cifar_classifier import NetworkInNetworkClassifier
from uncertainty.uq_through_redundancy.multi_input_classifier import MultiInputClassifier
from shared.data.dataset_meta import DatasetMeta as M


class EnsembleClassifier(ANN):
    """
    A wrapper for MultiInputClassifier for making predictions with multiple models.
    """
    WARNED = False

    def __init__(self, h: dict):
        super().__init__()
        self.h = h
        self.models: List[MultiInputClassifier] = []
        assert 'num_models' in h, "num_models must be specified in the hyperparameters."

        # assert 'checkpoint_path' contains h['num_models'] models
        num_models = h['num_models']
        checkpoint_path = h['checkpoint_path']
        assert os.path.isdir(checkpoint_path), f"checkpoint_path {checkpoint_path} does not exist."
        assert len(os.listdir(
            checkpoint_path)) == num_models, \
            (f"checkpoint_path {checkpoint_path} does not contain {num_models} models. "
             f"{len(os.listdir(checkpoint_path))} files were counted (the folder can only contain models).")

        self.create_classifiers(h)

        # loading weights still needs to be done manually using self.load()

    def forward(self, x: torch.Tensor):
        return self.forward_softmaxed(x)  # SOFTMAXED!!

    def _forward_multiple_models(self, x: torch.Tensor):
        """
        In: (batch, num_views, channels, height, width)
        Out: (batch, num_models, num_classes) # logits so softmax is not yet applied
        """

        y_hats = [model.forward(x)[0] for model in self.models]  # [(batch, nb_views, num_classes), ...]
        y_hats = torch.stack(y_hats, dim=1)  # (batch, num_models, nb_views, num_classes)

        # rm nb_views (assume 1)
        y_hats = y_hats.squeeze(2)
        return y_hats

    def forward_softmaxed(self, x: torch.Tensor):
        """
        In: (batch, num_views, channels, height, width)
        Out: (batch, num_classes)
        """
        if not self.WARNED:
            print("Warning: EnsembleClassifier.forward() returns softmax probabilities, not logits unlike other models."
                  "This warning will only be shown once.")
            self.WARNED = True

        y_hat = self._forward_multiple_models(x)  # (batch, num_models, num_classes)

        # Softmax before averaging!
        y_hat = torch.nn.functional.softmax(y_hat, dim=-1)  # (batch, num_models, num_classes)

        # Collapse the num_models dimension
        y_hat = y_hat.mean(dim=1)  # (batch, num_classes)
        return y_hat, None

    def load(self) -> 'EnsembleClassifier':
        assert self.h['model_name'] is None, \
            "model_name must be None when loading in an ensemble model as it has no effect."

        for i, model in enumerate(self.models):
            model: MultiInputClassifier  # type hint

            # Each model has a unique name, e.g. model_0.pth, model_1.pth, etc.
            name = f'model_{i}.pth'
            model.h['model_name'] = name
            model.load()  # load the model at `checkpoint_path` with the name `model_i.pth`

        return self

    @staticmethod
    def create_instance(h: dict) -> 'EnsembleClassifier':
        return EnsembleClassifier(h)

    def to(self, device):
        for model in self.models:
            model.to(device)

    @property
    def device(self):
        return self.models[0].device

    def save(self, name: Optional[str] = 'model.pth') -> None:
        raise NotImplementedError(
            "Ensemble models are not saved as a single model, but as multiple models. Use MultiInputClassifier.save instead.")

    def configure_optimizers(self):
        raise NotImplementedError("Ensemble models do not have an optimizer as they don't support training.")

    def training_step(self, batch, batch_idx):
        raise NotImplementedError("Ensemble models do not support training.")

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError("Ensemble models do not support validation.")

    def test_step(self, batch, batch_idx):
        raise NotImplementedError("Ensemble models do not support testing.")

    def create_classifiers(self, h: dict):
        for i in range(h['num_models']):
            if M.is_vision_dataset(h):
                model = NetworkInNetworkClassifier.create_instance(h)
            else:  # Radio dataset
                model = MultiInputClassifier.create_instance(h)
            self.models.append(model)
