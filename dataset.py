import torch
import os
import numpy as np
from typing import Tuple, List
import abc


def rgb2gray(img):
    return np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])


class CustomDataset(abc.ABC, torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_path: str,
        limit: int = None,
        preload_to_RAM: bool = False,
        crop_size: Tuple[int, int] = None,
        sequences: List[str] = [],
        convert_to_bw: bool = False,
        events_normalization: str = None,
    ):
        """
        Custom dataset that offers some utilities.

        Args:
            dataset_path (str): parent folder that contains one folder for each sequence.
            limit (int, optional): load only this number of samples. If None, no limit is imposed. Defaults to None.
            preload_to_RAM (bool, optional): preload samples to RAM, preprocessing them. Defaults to False.
            crop_size (Tuple[int, int], optional): crop events and images to this size (w x h). Defaults to None.
            sequences (List[str], optional): load only this list of sequences (folders). If empty list, every sequence will be loaded. Defaults to [].
            convert_to_bw (bool, optional): convert the ground truth images to black and white. Defaults to False.
            events_normalization (str, optional): normalize the events after being cropped.
            If None, no normalization will be perfomed, otherwise it should be one of the following strings:
            "min_max_0_1": min max normalization with values in the range [0, 1] (the max assumed is 255 and the min -255).
            "min_max_-1_1": min max normalization with values in the range [-1, 1] (the max assumed is 255 and the min -255).
            "z_score": z score normalization making values with mean 0 and std 1.
            "z_score_non_zero": z score normalization making values with mean 0 and std 1 only for values different from zero.
            Defaults to None.
        """
        super().__init__()
        self.dataset_path = dataset_path
        self.limit = limit
        self.preload_to_RAM = preload_to_RAM
        self.crop_size = crop_size
        self.convert_to_bw = convert_to_bw
        self.events_normalization = events_normalization

        self.files_list = []
        self.data = []

        if not sequences:
            sequences = os.listdir(dataset_path)

        for sequence_folder in sorted(sequences):
            seq_path = os.path.join(dataset_path, sequence_folder)

            # Support an additional folder called batches inside the sequence oflder
            if os.path.exists(os.path.join(seq_path, "batches")):
                seq_path = os.path.join(seq_path, "batches")

            for file in os.listdir(seq_path):
                if limit and len(self.files_list) >= limit:
                    break
                if file.endswith(".pt") or file.endswith(".npz"):
                    file_path = os.path.join(seq_path, file)
                    self.files_list.append(file_path)

                    if self.preload_to_RAM:
                        batch = self._load_batch(file_path)
                        sample = self.pre_process(batch)
                        self.data.append(sample)

    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, idx: int):
        if self.preload_to_RAM:
            sample = self.data[idx]
        else:
            file = self.files_list[idx]
            sample = self._load_batch(file)
            sample = self.pre_process(sample)
        return sample

    def _load_batch(self, file_path):
        if file_path.endswith(".pt"):
            return torch.load(file_path)
        elif file_path.endswith(".npz"):
            data = np.load(file_path)
            return [data["arr_" + str(i)] for i in range(len(data))]
        else:
            raise Exception(
                "File {} not supported. Only .pt and .npz files are supported.".format(
                    file_path
                )
            )

    def _normalize_events(self, events: torch.Tensor, norm_type: str):
        supported_types = ["min_max_0_1", "min_max_-1_1", "z_score", "z_score_non_zero"]
        if norm_type not in supported_types:
            raise Exception(
                'Normalization type "{}" not supported. You can choose from {}.'.format(
                    norm_type, supported_types
                )
            )

        # Define min and max for minmax
        e_min = -255
        e_max = 255

        if norm_type == "min_max_0_1":
            events = (events - e_min) / (e_max - e_min)
        elif norm_type == "min_max_-1_1":
            events = 2 * ((events - e_min) / (e_max - e_min)) - 1
        elif norm_type == "z_score":
            events = (events - events.mean()) / events.std()
        elif norm_type == "z_score_non_zero":
            nonzero_ev = events != 0
            num_nonzeros = nonzero_ev.sum()
            if num_nonzeros > 0:
                mean = events.sum() / num_nonzeros
                stddev = np.sqrt((events**2).sum() / num_nonzeros - mean**2)
                mask = nonzero_ev.astype(np.float32)
                events = mask * (events - mean) / stddev
        return events

    @abc.abstractmethod
    def pre_process(self, batch):
        pass


class CEDDataset(CustomDataset):
    def __init__(self, *args, ignore_input_image: bool = False, **kwargs):
        self.ignore_input_image = ignore_input_image
        super().__init__(*args, **kwargs)

    def pre_process(self, batch):
        (in_img, events), out_img = batch

        if self.crop_size:
            w, h = self.crop_size
            in_img = in_img[:h, :w, :]
            out_img = out_img[:h, :w, :]
            events = events[:, :h, :w]

        if self.events_normalization:
            events = self._normalize_events(events, self.events_normalization)

        if self.convert_to_bw:
            in_img = np.expand_dims(rgb2gray(in_img), -1)
            out_img = np.expand_dims(rgb2gray(out_img), -1)

        in_img = (in_img / 255.0).astype(np.float32)
        out_img = (out_img / 255.0).astype(np.float32)
        events = events.astype(np.float32)

        if self.ignore_input_image:
            return events, out_img
        else:
            return (in_img, events), out_img


class DIV2KDataset(CustomDataset):
    def pre_process(self, batch):
        events, out_img = batch

        if self.crop_size:
            w, h = self.crop_size
            out_img = out_img[:h, :w, :]
            events = events[:, :h, :w]

        if self.events_normalization:
            events = self._normalize_events(events, self.events_normalization)

        if self.convert_to_bw:
            out_img = np.expand_dims(rgb2gray(out_img), -1)

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
