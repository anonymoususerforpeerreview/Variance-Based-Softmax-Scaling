import torch
from torch.utils.data import DataLoader
from typing import Tuple

from shared.data.CustomDataLoader import CustomDataLoader
from shared.data.DatasetWithLazyTransform import DatasetWithLazyTransform


class RADIO(CustomDataLoader):
    _X_data = {}
    _Y_data = {}

    @staticmethod
    def _load_data_and_label(h: dict, path, batch_size, data_type, num_workers: int,
                             transformations: list) -> DataLoader:
        if data_type not in RADIO._X_data:
            print(f"Info: start loading the {data_type} data")
            RADIO._X_data[data_type] = torch.load(f"{path}X_{data_type}.pth")  # (b, c, t)
            RADIO._Y_data[data_type] = torch.load(f"{path}y_{data_type}.pth")  # (b, 1)

            # remove first 256 samples
            if h['remove_first_n_samples'] > 0:
                RADIO._X_data[data_type] = RADIO._X_data[data_type][:, :, h['remove_first_n_samples']:]
                # RADIO._Y_data[data_type] = RADIO._Y_data[data_type][h['remove_first_n_samples']:, :]

            # Flatten the channel and time dimensions for min and max calculation
            flat_X_data = RADIO._X_data[data_type].reshape(RADIO._X_data[data_type].shape[0],
                                                           -1)  # Flattening to (b, c*t)

            # Calculate global min and max across both channels
            min_vals = flat_X_data.min(dim=1, keepdim=True)[0].view(-1, 1, 1)
            max_vals = flat_X_data.max(dim=1, keepdim=True)[0].view(-1, 1, 1)

            # Apply normalization using global min and max
            RADIO._X_data[data_type] = -1 + 2 * (RADIO._X_data[data_type] - min_vals) / (max_vals - min_vals)
            RADIO._X_data[data_type] = RADIO._X_data[data_type].float()  # (dataset_size, nb_channels, nb_timesteps)
            print(f"Info: loaded the {data_type} signals")

        # Apply transformations
        data_dataset = DatasetWithLazyTransform(h, transformations, RADIO._X_data[data_type], RADIO._Y_data[data_type])
        shuffle = True if data_type == 'train' else False
        data_loader = DataLoader(data_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True,
                                 num_workers=num_workers, persistent_workers=True)
        return data_loader

    @staticmethod
    def _get_radio_data_loaders(h: dict, dataset_path: str, batch_size: int, num_workers: int, transformations: list) -> \
            Tuple[DataLoader, DataLoader, DataLoader]:
        path = f"{dataset_path}/RadioIdentification/"
        train_loader: DataLoader = RADIO._load_data_and_label(h, path, batch_size, 'train', num_workers=num_workers,
                                                              transformations=transformations)

        val_loader = RADIO._load_data_and_label(h, path, batch_size, 'val', num_workers=num_workers,
                                                transformations=transformations)
        test_loader = RADIO._load_data_and_label(h, path, batch_size, 'test', num_workers=num_workers,
                                                 transformations=transformations)
        return train_loader, val_loader, test_loader

    @staticmethod
    def data_loaders(h: dict, transformations: list, dataset_path, batch_size, workers) -> Tuple[
        DataLoader, DataLoader, DataLoader]:
        print("Info: loading RADIO data")
        return RADIO._get_radio_data_loaders(h, dataset_path, batch_size, workers, transformations)
