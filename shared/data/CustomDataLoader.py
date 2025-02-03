from typing import Tuple
from torch.utils.data import DataLoader


class CustomDataLoader:
    @staticmethod
    def data_loaders(h: dict, transformations: list, dataset_path, batch_size, workers) -> Tuple[
        DataLoader, DataLoader, DataLoader]:
        raise NotImplementedError
