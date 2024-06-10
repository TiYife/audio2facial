import os

import torch
import argparse
import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    TQDMProgressBar as ProgressBar,
    # DeviceStatsMonitor,
)
from lightning.pytorch.loggers import TensorBoardLogger

from Nvidia2017.dataset_nv2017 import VocaDataModule
from Nvidia2017.lighting_model_nv_2017 import Audio2FaceModel
from utils.config import ExpConfig

if __name__ == "__main__":
    # torch.multiprocessing.set_start_method("spawn")
    torch.set_float32_matmul_precision("medium")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    # training parameters
    dataset_path = "./trainingdata"
    config = ExpConfig.from_yaml(args.config)

    is_transformer = config.modelname == "faceformer"
    if is_transformer:
        config.split_frame = False
        config.batch_size = 1
        config.feature_extractor = None

    voca_datamodule = VocaDataModule(
        dataset_path,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        split_frame=config.split_frame,
    )
    voca_datamodule.setup()
    config.n_max_sentences = voca_datamodule.get_max_sentence_num() + 1
    config.n_max_frames = voca_datamodule.get_max_frame_num() + 1

    version = config.name()

    # Train
    model = Audio2FaceModel(config)

    trainer = L.Trainer(
        accelerator='gpu',
        devices=1,
        precision=config.percision,
        log_every_n_steps=10,
        # logger=TensorBoardLogger("logs", name=version),
        logger=False,
        callbacks=[
            ModelCheckpoint(monitor="val/err", save_last=True),
            EarlyStopping(monitor="val/err", patience=5),
            ProgressBar(),
            # DeviceStatsMonitor(),
        ],
        max_epochs=50,
    )
    trainer.fit(model, datamodule=voca_datamodule)

    # ckpts = os.listdir(trainer.log_dir + "/checkpoints")
    ckpts = os.listdir("E:/Audio2Facial/audio2facial/logs/audio2mesh_mfcc_0.0001_None_16-mixed/version_92/checkpoints")
    sorted_ckpts = sorted(ckpts, key=lambda x: int(x.split("=")[-1].split(".")[0]))

    model = Audio2FaceModel.load_from_checkpoint(
        # trainer.log_dir + "/checkpoints/" + sorted_ckpts[-1]
        "E:/Audio2Facial/audio2facial/logs/audio2mesh_mfcc_0.0001_None_16-mixed/version_92/checkpoints/" + sorted_ckpts[-1]
    )

    # inference only
    # trainer = L.Trainer()
    # voca_datamodule.setup("test")

    trainer.predict(
        model,
        voca_datamodule.predict_dataloader("FaceTalk_170908_03277_TA", "sentence02"),
    )
