from pydantic import BaseModel
from typing import Optional
import yaml


class ExpConfig(BaseModel):
    # dataset
    batch_size: int
    num_workers: int
    # model
    modelname: str
    one_hot_size: int
    feature_extractor: str
    sample_rate: int
    vertex_count: int
    split_frame: bool
    n_feature: int
    out_dim: int
    win_length: int
    hop_length: Optional[int] = None
    # training
    percision: str
    lr: float
    # loss
    loss: Optional[str] = None

    # emotion
    enable_emotions: bool = False
    n_max_sentences: Optional[int] = None
    n_max_frames: Optional[int] = None
    n_emotion_dim: Optional[int] = None

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        return cls(**config)

    def name(self):
        return f"{self.modelname}_{self.feature_extractor}_{self.lr}_{self.loss}_{self.percision}"
