from typing import Tuple

import lightning as L
import torch


class DataModule(L.LightningDataModule):
    def __init__(self, train_loader, val_loader, test_loader):
        super().__init__()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return self.test_loader

    def get_all_data(self, loader: torch.utils.data.DataLoader, subset_percentage: float) -> Tuple[
        torch.Tensor, torch.Tensor]:
        assert subset_percentage > 0 and subset_percentage <= 1, \
            "subset_percentage must be between 0 and 1."

        inputs_list = torch.Tensor()
        labels_list = torch.Tensor()
        first_batch = True

        for inputs, labels in loader:
            if first_batch:
                inputs_list = inputs
                labels_list = labels
                first_batch = False
            else:
                inputs_list = torch.cat((inputs_list, inputs), dim=0)
                labels_list = torch.cat((labels_list, labels), dim=0)

        # Calculate the number of samples to include in the subset
        total_samples = inputs_list.shape[0]
        subset_size = int(total_samples * subset_percentage)

        # Generate random indices for the subset
        indices = torch.randperm(total_samples)[:subset_size]

        # Index into inputs_list and labels_list to get the subset
        inputs_subset = inputs_list[indices]
        labels_subset = labels_list[indices]

        print("-" * 10)
        print(
            f"In get_all_data: inputs_subset.shape = {inputs_subset.shape}, labels_subset.shape = {labels_subset.shape}")
        print("-" * 10)

        return inputs_subset, labels_subset
