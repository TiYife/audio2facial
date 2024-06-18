import argparse
import os
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from Nvidia2017.dataset_nv2017 import VocaDataModule
from Nvidia2017.lighting_model_nv_2017 import Audio2FaceModel
from util.config import ExpConfig

# 创建模型对象
model = Audio2FaceModel.load_from_checkpoint("path/to/your/checkpoint.ckpt")

# 创建 Trainer 对象
trainer = Trainer()

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="config.yaml")
args = parser.parse_args()
config = ExpConfig.from_yaml(args.config)

dataset_path = "./trainingdata"
voca_datamodule = VocaDataModule(
    dataset_path,
    batch_size=config.batch_size,
    num_workers=config.num_workers,
    split_frame=config.split_frame,
)

# 设置数据模块
voca_datamodule.setup("test")

# 进行推断
predictions = trainer.predict(model, datamodule=voca_datamodule.predict_dataloader("FaceTalk_170908_03277_TA", "sentence02"))

# 处理预测结果
# 这里可以根据推断结果进行后续处理，例如保存、打印或其他操作
print(predictions)
