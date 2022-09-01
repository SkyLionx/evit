import torch
import os
import numpy as np
from typing import Tuple

class CEDDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path: str, limit: int = None, preload_to_RAM: bool = False, crop_size: Tuple[int, int] = None):
        self.dataset_path = dataset_path
        self.limit = limit
        self.preload_to_RAM = preload_to_RAM
        self.crop_size = crop_size
        self.files_list = []
        self.data = []

        for file in sorted(os.listdir(dataset_path)):
            if limit and len(self.files_list) >= limit:
              break
            if file.endswith(".pt"):
                file_path = os.path.join(dataset_path, file)
                self.files_list.append(file_path)
                if self.preload_to_RAM:
                    sample = self.pre_process(*torch.load(file_path))
                    self.data.append(sample)

    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, idx: int):
        file = self.files_list[idx]
        
        # Old dataset format
        # (in_img, events), out_img = torch.load(file)
        # in_img = (in_img[:256, :336, :] / 255.0).astype(np.float32)
        # out_img = (out_img[:256, :336, :] / 255.0).astype(np.float32)
        # return (in_img, events[:, :256, :336].astype(np.float32)), out_img

        if self.preload_to_RAM:
            events, out_img = self.data[idx]
        else:
            events, out_img = torch.load(file)
            events, out_img = self.pre_process(events, out_img)
        return events, out_img

    def pre_process(self, events, out_img):
        if self.crop_size:
            w, h = self.crop_size
            out_img = out_img[:h, :w, :]
            events = events[:, :h, :w]

        out_img = (out_img / 255.0).astype(np.float32)
        events = events.astype(np.float32)
        return events, out_img


class ConcatBatchSampler(torch.utils.data.Sampler):
    """
    Sampler that is aware of multiple samplers.
    It doesn't batch together samples coming from different samplers.
    """

    def __init__(self, samplers, batch_size: int, drop_last: bool) -> None:
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ValueError(
                "batch_size should be a positive integer value, "
                "but got batch_size={}".format(batch_size)
            )
        if not isinstance(drop_last, bool):
            raise ValueError(
                "drop_last should be a boolean value, but got "
                "drop_last={}".format(drop_last)
            )

        self.samplers = samplers
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        start_idx = 0
        if self.drop_last:
            for sampler in self.samplers:
                sampler_iter = iter(sampler)
                while True:
                    try:
                        batch = [next(sampler_iter) for _ in range(self.batch_size)]
                        yield batch
                    except StopIteration:
                        break
        else:
            for sampler in self.samplers:
                batch = [0] * self.batch_size
                idx_in_batch = 0
                for idx in sampler:
                    batch[idx_in_batch] = start_idx + idx
                    idx_in_batch += 1
                    if idx_in_batch == self.batch_size:
                        yield batch
                        idx_in_batch = 0
                        batch = [0] * self.batch_size
                if idx_in_batch > 0:
                    yield batch[:idx_in_batch]
                start_idx += len(sampler)

    def __len__(self) -> int:
        if self.drop_last:
            return sum([len(sampler) // self.batch_size for sampler in self.samplers])
        else:
            return sum(
                [
                    (len(sampler) + self.batch_size - 1) // self.batch_size
                    for sampler in self.samplers
                ]
            )
