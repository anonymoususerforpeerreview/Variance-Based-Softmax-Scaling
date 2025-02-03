import os
from abc import ABC, abstractmethod
from typing import Optional

import lightning as L
import torch
from torch import Tensor


def display_progress(h: dict, epoch: int, batch_idx: int, loss: float) -> None:
    if batch_idx == 0 and h['enable_progress_bar'] == False:
        print(f"Epoch {epoch} - Training loss {loss}.")


class ANN(L.LightningModule, ABC):
    def training_step(self, batch, batch_idx):
        x, y = batch  # x: (batch, nb_views, C, H, W), y: (batch)
        y_hat, latent_representations = self.forward(x)  # (batch, nb_views, num_classes)

        # assert y and y_hat have the correct shape
        assert y.dim() == 1, f"y shape is {y.shape}, expected 1 dimension."
        assert y_hat.dim() == 3, f"y_hat shape is {y_hat.shape}, expected 3 dimensions."

        loss = self.loss(y_hat, y, latent_representations, x.size(1))
        self.log('train_loss', loss, batch_size=x.size(0))

        # print epoch number at start of epoch when progress bar is disabled
        display_progress(self.h, self.current_epoch, batch_idx, loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat, latent_representations = self(x)

        loss = self.loss(y_hat, y, latent_representations, x.size(1))
        self.log('val_loss', loss, batch_size=x.size(0))
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch  # x: (batch, nb_views, C, H, W), y: (batch)
        y_hat, latent_representations = self.forward(x)  # y_hat: (batch, nb_views, num_classes)

        loss = self.loss(y_hat, y, latent_representations, x.size(1))

        y_hat_mean = y_hat.mean(dim=1)  # (batch, num_classes)
        acc = (y_hat_mean.argmax(dim=1) == y).float().mean()

        self.log('test_acc', acc)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.h['learning_rate'], weight_decay=self.h['weight_decay'])

    def save(self) -> None:

        if self.h['model_name'] is None:  # Default name
            name: Optional[str] = 'model.pth'
        else:
            # this can be used to create an ensemble of models.
            # Overwrite via hyperparameters and choose a naming scheme of "model_0.pth", "model_1.pth", etc.
            # They must be saved in the same directory.
            name = self.h['model_name']  # Custom name

        if not (name == 'model.pth'):
            print(f"Warning: model saved as `{name}`, make sure to load it in correctly as well then!")

        os.makedirs(self.h['checkpoint_path'], exist_ok=True)
        torch.save(self.state_dict(), os.path.join(self.h['checkpoint_path'], name))

        print(f"Model saved at {os.path.join(self.h['checkpoint_path'], name)}")

    def load(self) -> 'ANN':
        if self.h['model_name'] is None:  # Default name
            name: Optional[str] = 'model.pth'
        else:
            name = self.h['model_name']

        self.load_state_dict(torch.load(os.path.join(self.h['checkpoint_path'], name)))
        self.eval()
        print(f"Model loaded from {os.path.join(self.h['checkpoint_path'], name)}")
        return self

    @abstractmethod
    def forward(self, x: Tensor):
        pass

    @staticmethod
    @abstractmethod
    def create_instance(h: dict) -> 'ANN':
        pass
