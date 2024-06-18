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


class FlameModelConfig(BaseModel):
    flame_model_path: str = "./assets/flame/flame2020/generic_model.pkl"
    static_landmark_embedding_path: str = "./assets/flame/flame_static_embedding.pkl"
    dynamic_landmark_embedding_path: str = "./assets/flame/flame_dynamic_embedding.npy"
    shape_params: int = 300
    expression_params: int = 100
    pose_params: int = 6
    use_face_contour: bool = True
    use_3D_translation: bool = True
    optimize_eyeballpose: bool = True
    optimize_neckpose: bool = True
    num_worker: int = 4
    batch_size: int = 1
    ring_margin: float = 0.5
    ring_loss_weight: float = 1.0

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        return cls(**config)


def get_flame_config():
    # 创建配置对象并返回
    config = FlameModelConfig()
    return config
