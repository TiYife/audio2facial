import os
import torch
import argparse
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from Nvidia2017.dataset_nv2017 import VocaDataModule
from Nvidia2017.lighting_model_nv_2017 import Audio2FaceModel
from utils.config import ExpConfig

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    # 加载配置
    config = ExpConfig.from_yaml(args.config)

    # 初始化数据模块
    dataset_path = "./testdata"  # 测试数据路径
    voca_datamodule = VocaDataModule(
        dataset_path,
        batch_size=1,  # 设置为1，一个样本进行预测
        num_workers=config.num_workers,
        split_frame=False,  # 在预测时不需要分帧
    )
    voca_datamodule.setup("test")

    # 选择具有最大 id 的输出文件夹下的检查点文件
    output_dir = "path/to/output"  # 输出文件夹路径
    checkpoint_dir = max([os.path.join(output_dir, d) for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))])
    checkpoint_file = max([os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.endswith(".ckpt")])

    # 加载模型
    model = Audio2FaceModel.load_from_checkpoint(checkpoint_file)

    # 创建 Trainer 对象
    trainer = L.Trainer()

    # 进行预测
    predictions = trainer.predict(model, datamodule=voca_datamodule)

    # 处理预测结果
    print(predictions)
