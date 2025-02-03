import glob
import os
from typing import List
import torch
from torch import Tensor
import uuid


class ChunkWriter:
    def __init__(self):
        self.unique_id = str(uuid.uuid4())

    def save_chunks(self, tensor_list: List[Tensor], chunk_size: int, prefix: str):
        for i in range(0, len(tensor_list), chunk_size):
            chunk = torch.cat(tensor_list[i:i + chunk_size], dim=0)
            torch.save(chunk, f"{prefix}_{self.unique_id}_chunk_{i // chunk_size}.pt")

    def load_and_delete_chunks(self, prefix: str) -> Tensor:
        all_chunks = []
        for chunk_file in sorted(glob.glob(f"{prefix}*_{self.unique_id}_chunk_*.pt")):
            all_chunks.append(torch.load(chunk_file))
            os.remove(chunk_file)
        return torch.cat(all_chunks, dim=0)
