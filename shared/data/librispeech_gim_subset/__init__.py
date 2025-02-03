import os
import os.path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, Optional

import torch
import torchaudio
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from shared.data.CustomDataLoader import CustomDataLoader
from shared.data.transformations import T


def default_flist_reader(flist):
    item_list = []
    speaker_dict = defaultdict(list)
    index = 0
    with open(flist, "r") as rf:
        for line in rf.readlines():
            speaker_id, dir_id, sample_id = line.replace("\n", "").split("-")
            item_list.append((speaker_id, dir_id, sample_id))
            speaker_dict[speaker_id].append(index)
            index += 1

    return item_list, speaker_dict


def build_speaker_id_dict(train_file_list, test_file_list):
    speaker_dict = defaultdict(list)
    index = 0
    for file_list in [train_file_list, test_file_list]:
        for speaker_id, _, _ in file_list:
            if speaker_id not in speaker_dict:
                speaker_dict[speaker_id] = index
                index += 1
    return speaker_dict


class LibriDataset(Dataset):
    def __init__(self, h: dict, transformations: list[callable], root, flist, speaker_id_dict, audio_length,
                 speaker_ids=None, flist_reader=default_flist_reader):
        self.root = root
        self.audio_length = audio_length
        self.h: dict = h
        self.transformations = transformations
        self.speaker_id_dict = speaker_id_dict

        assert not (h["redundancy_method"] == "identity"), \
            ("Identity redundancy method is not supported for LibriSpeech dataset. "
             "The transformation function should crop the data to fixed length (so that batch is consistent).")

        # Read file list and filter out short audio files
        self.file_list, _ = flist_reader(flist)
        if speaker_ids:  # Only keep the specified speaker IDs (used for testing)
            self.file_list = [file_info for file_info in self.file_list if file_info[0] in speaker_ids]
        self.file_list = self._filter_short_audio_files(self.file_list)
        self.speaker_ids = {speaker_id for speaker_id, _, _ in self.file_list}
        print(f"Preserved {len(self.file_list) / len(flist_reader(flist)[0]) * 100:.2f}% of the dataset. "
              f"Audio files: {audio_length / 16000:.2f} seconds long.")

    def _filter_short_audio_files(self, file_list):
        def is_valid_audio(file_info):
            speaker_id, dir_id, sample_id = file_info
            filename = f"{speaker_id}-{dir_id}-{sample_id}"
            file_path = os.path.join(self.root, speaker_id, dir_id, f"{filename}.flac")
            audio, samplerate = torchaudio.load(file_path)
            return audio.size(1) >= self.audio_length

        with ThreadPoolExecutor() as executor:
            filtered_list = list(executor.map(is_valid_audio, file_list))

        return [file_info for file_info, is_valid in zip(file_list, filtered_list) if is_valid]

    def __getitem__(self, index):
        speaker_id, dir_id, sample_id = self.file_list[index]
        filename = f"{speaker_id}-{dir_id}-{sample_id}"
        file_path = os.path.join(self.root, speaker_id, dir_id, f"{filename}.flac")

        audio, samplerate = torchaudio.load(file_path)
        assert samplerate == 16000, "Watch out, samplerate is not consistent throughout the dataset!"

        # discard last part that is not a full 10ms
        max_length = audio.size(1) // 160 * 160
        audio = audio[:max_length]

        if self.transformations:  # T.create_multiple_views
            for t in self.transformations:
                audio = t(audio, self.h)  # h is passed to the transformation

        target = self.speaker_id_dict[speaker_id]
        return audio, target

    def __len__(self):
        return len(self.file_list)


class LIBRISPEECH_GIM_SUBSET(CustomDataLoader):
    @staticmethod
    def data_loaders(h: dict, transformations: list, dataset_path, batch_size, workers) -> Tuple[
        DataLoader, Optional[DataLoader], DataLoader]:
        print("Info: loading LIBRISPEECH data")
        root = os.path.join(dataset_path, "LibriSpeech/train-clean-100")

        train_file_list, _ = default_flist_reader(f"{dataset_path}/LibriSpeech100_labels_split/train_split.txt")
        test_file_list, _ = default_flist_reader(f"{dataset_path}/LibriSpeech100_labels_split/test_split.txt")

        speaker_id_dict = build_speaker_id_dict(train_file_list, test_file_list)

        redun_method = h['redundancy_method']
        crop_size = h['redundancy_method_params']['crop_size']
        assert redun_method, "Only random crop redundancy method is supported at the moment."
        assert crop_size % 160 == 0, "Crop size must be divisible by 160"

        if h["use_only_k_labels"] > 0:
            # Count occurrences of each speaker ID in train and test file lists
            speaker_counts = defaultdict(int)
            for file_list in [train_file_list, test_file_list]:
                for speaker_id, _, _ in file_list:
                    speaker_counts[speaker_id] += 1

            # Select the top 10 most frequent speaker IDs
            top_speakers = sorted(speaker_counts, key=speaker_counts.get, reverse=True)[:10]

            # overwrite speaker_id_dict with top_speakers (change into values 0-9)
            speaker_id_dict = {speaker_id: index for index, speaker_id in enumerate(top_speakers)}
        else: # all speakers
            top_speakers = None

        train_dataset = LibriDataset(h, transformations, root,
                                     f"{dataset_path}/LibriSpeech100_labels_split/train_split.txt",
                                     speaker_id_dict, audio_length=crop_size, speaker_ids=top_speakers)

        test_dataset = LibriDataset(h, transformations, root,
                                    f"{dataset_path}/LibriSpeech100_labels_split/test_split.txt",
                                    speaker_id_dict, audio_length=crop_size, speaker_ids=top_speakers)

        # Split the test dataset into test and validation sets
        test_size = len(test_dataset) // 2
        val_size = len(test_dataset) - test_size
        test_dataset, val_dataset = torch.utils.data.random_split(test_dataset, [test_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers,
                                  persistent_workers=True, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                                 persistent_workers=True, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                                persistent_workers=True, pin_memory=True)

        if h["use_only_k_labels"] > 0:
            # Print chosen speaker IDs
            assert sorted(train_dataset.speaker_ids) == [
                '118', '125', '211', '27', '2989', '4014', '4195', '5339', '730', '8063'], "Unexpected speaker IDs"
            print(f"Train speaker IDs: {sorted(train_dataset.speaker_ids)}")
            print(f"Test speaker IDs: {sorted(test_dataset.dataset.speaker_ids)}")
            print(f"Validation speaker IDs: {sorted(val_dataset.dataset.speaker_ids)}")

        # Print length of speaker ids
        print(f"Train speaker IDs: {len(train_dataset.speaker_ids)}, Test speaker IDs: "
              f"{len(test_dataset.dataset.speaker_ids)}, Validation speaker IDs: {len(val_dataset.dataset.speaker_ids)}")

        return train_loader, val_loader, test_loader


if __name__ == "__main__":
    h = {
        "batch_size": 4,
        "workers": 2,
        # "redundancy_method": "identity",
        # "redundancy_method_params": {},
        "redundancy_method": "random_crop",
        "redundancy_method_params": {"crop_size": 20480},  # *8
        "num_views": 1,
        "use_only_k_labels": False, # ony 10 speakers
        # --redundancy_method
        # random_crop
        # --redundancy_method_params
        # {'crop_size':20480}
    }
    transformations = [T.create_multiple_views]
    dataset_path = "./../../../../Smooth-InfoMax\datasets"
    batch_size = 4
    workers = 2

    # train_loader, val_loader, test_loader = LIBRISPEECH_GIM_SUBSET.data_loaders(h, transformations, dataset_path, batch_size,
    #                                                                    workers)
    #
    # for (x, y) in train_loader:
    #     print(x.shape)  # torch.Size([4, 1, 1, 20480])
    #     break
    #
    # for (x, y) in test_loader:
    #     print(x.shape)
    #     break

    train_loader, val_loader, test_loader = LIBRISPEECH_GIM_SUBSET.data_loaders(h, transformations, dataset_path,
                                                                                batch_size, workers)

    train_count = sum(len(x) for x, y in train_loader)
    val_count = sum(len(x) for x, y in val_loader)
    test_count = sum(len(x) for x, y in test_loader)

    print(f"Number of examples in train loader: {train_count}")
    print(f"Number of examples in validation loader: {val_count}")
    print(f"Number of examples in test loader: {test_count}")
