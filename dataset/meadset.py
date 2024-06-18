import os
from typing import TypedDict, Mapping, Literal, List

import torch
import torchaudio
import random
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import lightning as L
from rich import print

from dataset.utils import load_pickle, load_npy_mmapped


class MeadItem(TypedDict):
    audio: torch.Tensor
    vertexes: torch.Tensor
    template: torch.Tensor
    human_one_hot: torch.Tensor
    emotion_one_hot: torch.Tensor
    emotion_intensity: torch.Tensor


class MeadSplitRecorder:
    def __init__(self, write: bool = True):
        self.write = write
        self.train_list = []
        self.val_list = []
        self.test_list = []

    def can_write(func):
        def wrapp(self, *args, **kwargs):
            if self.write:
                return func(self, *args, **kwargs)
            else:
                raise ValueError("This instance is read only")

        return wrapp

    @can_write
    def split(self, all_keys: List, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        random.shuffle(all_keys)

        total_size = len(all_keys)
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)

        self.train_list = all_keys[:train_size]
        self.val_list = all_keys[train_size:train_size + val_size]
        self.test_list = all_keys[train_size + val_size:]

    @can_write
    def save(self, path: str):
        def _save_list(ls, name):
            df = pd.DataFrame(
                ls,
                columns=["human_id", "emotion_id", "emotion_level", "content_id"],
            )
            df.to_csv(name, index=False)

        _save_list(self.train_list, f"{path}/train_list.csv")
        _save_list(self.val_list, f"{path}/val_list.csv")
        _save_list(self.test_list, f"{path}/test_list.csv")

    @staticmethod
    def build(
            all_keys: List,
            save_path: str = None,
    ):
        save_path = os.path.join(save_path, "split")
        os.makedirs(save_path, exist_ok=False)
        data_split_recorder = MeadSplitRecorder()
        data_split_recorder.split(all_keys)

        data_split_recorder.save(save_path)

    @staticmethod
    def exists(path: str):
        path = os.path.join(path, "split")
        return (
                os.path.exists(f"{path}/train_list.csv")
                and os.path.exists(f"{path}/val_list.csv")
                and os.path.exists(f"{path}/test_list.csv")
        )

    @classmethod
    def load(cls, path: str):
        path = os.path.join(path, "split")

        def _load_list(name):
            df = pd.read_csv(name, header=0)
            return df.to_numpy().tolist()

        train_list = _load_list(f"{path}/train_list.csv")
        val_list = _load_list(f"{path}/val_list.csv")
        test_list = _load_list(f"{path}/test_list.csv")
        recorder = cls(write=False)
        recorder.train_list = train_list
        recorder.val_list = val_list
        recorder.test_list = test_list
        return recorder

    def get_list(self, phase: Literal["train", "val", "test", "all"] = "all"):
        if phase == "train":
            return self.train_list
        elif phase == "val":
            return self.val_list
        elif phase == "test":
            return self.test_list
        else:
            return self.train_list + self.val_list + self.test_list


class MeadSet(Dataset):
    def __init__(self,
                datapath: str,
                phase: Literal["train", "val", "test"],
                random_shift: bool = False, ):

        self.datapath = datapath
        self.phase = phase
        self.random_shift = random_shift

        self.all_keys = load_npy_mmapped(os.path.join(datapath, "all_keys.npy"))
        self.audios = load_npy_mmapped(os.path.join(datapath, "raw_audio.npy"))
        self.verts = load_pickle(os.path.join(datapath, "raw_vertex.pkl"))
        self.temps = load_pickle(os.path.join(datapath, "raw_template.pkl"))

        self.all_human_ids = self.collect_all_human_ids()

        if not MeadSplitRecorder.exists(datapath):
            print("Splitting mead dataset")
            MeadSplitRecorder.build(self.all_keys, datapath)

        self.data_spliter = MeadSplitRecorder.load(datapath)
        self.keys = self.data_spliter.get_list(phase)
        self.keys_len = len(self.keys)
        print(f"Loaded Total {phase} keys: {self.keys_len}")

    def __repr__(self):
        return f"{self.__class__.__name__}({self.datapath}, {self.phase}, {self.keys_len})"

    def __len__(self):
        return self.keys_len

    def __getitem__(self, item):
        key = self.keys[item]

        audio = self.audios[key]
        verts = self.verts[key]
        template = self.temps[key]

        # todo return other data
        human_id, emotion_id, emotion_level, content_id = key

        return MeadItem(
            audio=audio,
            vertexes=verts,
            template=template,
            human_one_hot=self.get_human_id_one_hot(human_id),
        )

    def collect_all_human_ids(self):
        all_human_ids = set()
        for key in self.all_keys:
            human_id, emotion_id, emotion_level, content_id = key
            all_human_ids.add(human_id)

        return all_human_ids

    def get_human_id_one_hot(self, human_id):
        one_hot = np.zeros(len(self.all_human_ids))
        one_hot[self.all_human_ids.index(human_id)] = 1
        return one_hot


class MeadDataModule(L.LightningDataModule):
    def __init__(self,
                 datapath: str,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 random_shift: bool = False,
                 ):
        super().__init__()
        self.datapath = datapath
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.random_shift = random_shift

    def setup(self, stage: str = None):
        self.train_dataset = MeadSet(
            self.datapath,
            phase="train",
            random_shift=self.random_shift
            )
        self.val_dataset = MeadSet(
            self.datapath,
            phase="val",
            random_shift=self.random_shift
            )
        self.test_dataset = MeadSet(
            self.datapath,
            phase="test",
            random_shift=self.random_shift
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            )

    def predict_dataloader(self):
        # todo implement
        pass