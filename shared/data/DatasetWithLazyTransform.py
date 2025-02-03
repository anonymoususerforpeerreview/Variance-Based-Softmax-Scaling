import torch
from torch.utils.data import TensorDataset


class DatasetWithLazyTransform(TensorDataset):
    def __init__(self, h: dict, transformations: list[callable], tensors: torch.tensor, targets: torch.tensor):
        super(DatasetWithLazyTransform, self).__init__(tensors, targets)
        self.h: dict = h
        self.transformations = transformations
        self.tensors = tensors
        self.targets = targets

    def __getitem__(self, index):
        # IDK what this is for, copilot suggested it, but i think it solved nothing
        x = self.tensors[index]
        y = self.targets[index]

        if self.transformations:  # T.create_multiple_views
            for t in self.transformations:
                x = t(x, self.h)  # h is passed to the transformation

        return x, y

    def __len__(self):
        return len(self.tensors)
