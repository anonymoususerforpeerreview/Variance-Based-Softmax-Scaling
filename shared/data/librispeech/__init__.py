import numpy as np
from typing import Tuple

import torchaudio
from torch.utils.data import DataLoader
from torchaudio.datasets import LIBRISPEECH

from shared.data.CustomDataLoader import CustomDataLoader
from shared.data.transformations import T
from shared.hyperparameters import Hyperparameters


class LibriSpeechDataset(LIBRISPEECH):
    def __init__(self, h: dict, root, url, folder_in_archive='LibriSpeech', download=False,
                 transformations: list[callable] = None):
        super(LibriSpeechDataset, self).__init__(root, url, folder_in_archive, download)
        self.h = h
        self.transformations: list[callable] = transformations
        self.audio_length = 20480

    def __getitem__(self, index):
        raise NotImplementedError(
            "here we still return speaker_id, but in gim_subset we return target label based on a dictionary "
            "e.g.: speaker_id = '2345' -> int \in [0, 251]. Maybe this code should be changed similarly.")
        waveform, sample_rate, utterance, speaker_id, chapter_id, utterance_id = \
            super(LibriSpeechDataset, self).__getitem__(index)

        # Take a crop of 20480 samples
        max_length = waveform.size(1) // 160 * 160
        start_idx = np.random.choice(
            np.arange(160, max_length - self.audio_length - 0, 160)
        )
        waveform = waveform[:, start_idx: start_idx + self.audio_length]

        if self.transformations:  # T.create_multiple_views
            for t in self.transformations:
                waveform = t(waveform, self.h)  # h is passed to the transformation

        assert sample_rate == 16000, "Watch out, samplerate is not consistent throughout the dataset!"

        return waveform, speaker_id


# def collate_fn(batch):
#     waveforms, speaker_ids = zip(*batch)
#     return waveforms, speaker_ids


def get_librispeech_dataloader(h: dict, root, url, batch_size, num_workers, transformations: list[callable],
                               shuffle=True):
    dataset = LibriSpeechDataset(h, root, url, download=True, transformations=transformations)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                            pin_memory=True, persistent_workers=True)
    return dataloader


class LIBRISPEECH(CustomDataLoader):
    @staticmethod
    def data_loaders(h: dict, transformations: list, dataset_path, batch_size, workers) -> Tuple[
        DataLoader, DataLoader, DataLoader]:
        print("Info: loading LIBRISPEECH data")
        return LIBRISPEECH._data_loaders(h, dataset_path, batch_size, workers, transformations)

    @staticmethod
    def _data_loaders(h: dict, dataset_path: str, batch_size: int, num_workers: int, transformations: list) -> Tuple[
        DataLoader, DataLoader, DataLoader]:
        train_loader = get_librispeech_dataloader(h, dataset_path, "train-clean-100", batch_size, num_workers,
                                                  transformations=transformations, shuffle=True)
        val_loader = get_librispeech_dataloader(h, dataset_path, "dev-clean", batch_size, num_workers,
                                                transformations=transformations, shuffle=False)
        test_loader = get_librispeech_dataloader(h, dataset_path, "test-clean", batch_size, num_workers,
                                                 transformations=transformations, shuffle=False)
        return train_loader, val_loader, test_loader


if __name__ == "__main__":
    root = "./../Smooth-InfoMax\datasets"  # assumes we execute this script from the root of the project
    # url = "train-clean-100"
    url = "dev-clean"
    batch_size = 4
    num_workers = 2
    transform = [T.create_multiple_views]  # no transformation
    h = Hyperparameters.get()

    # h: dict, root, url, batch_size, num_workers, transformations: list[callable], shuffle=True
    dataloader = get_librispeech_dataloader(h, root, url, batch_size, num_workers, transformations=transform)

    for batch in dataloader:
        waveforms, speaker_ids = batch
        print(waveforms)
        print(speaker_ids)

        # store a waveform to temp.wav
        torchaudio.save("temp.wav", waveforms[0], 16_000)

        break
