import os
from typing import TypedDict

import librosa
import pickle
import h5py
import numpy as np
import torch
import warnings

from dataset.utils import save_npy_mmapped, save_pickle
from model.flame import get_default_flame

warnings.filterwarnings('ignore')


def load_audio(audio_dir):
    audio_dict = {}
    for root, dirs, files in os.walk(audio_dir):
        for file in files:
            if file.endswith(".m4a"):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, audio_dir)

                human_id, emotion_id, emotion_level = relative_path.split(os.sep)[:-1]
                content_id = os.path.splitext(file)[0]
                print(human_id, emotion_id, emotion_level, content_id)

                audio, sr = librosa.load(file_path)
                audio_dict[(human_id, emotion_id, emotion_level, content_id)] = audio
    return audio_dict


def load_vertexes(vertex_dir):
    vertex_dict = {}
    template_dict = {}
    flame = get_default_flame()
    for root, dirs, files in os.walk(vertex_dir):
        for file in files:
            if file == "shape_pose_cam.hdf5":
                file_path = os.path.join(root, file)

                data_dict = {}
                with h5py.File(file_path, "r") as f:
                    for k in f.keys():
                        print(f"shape of {k} : {f[k].shape}")
                        data_dict[k] = torch.from_numpy(f[k][:])

                shape = data_dict['shape'].squeeze(0)
                exp = data_dict['exp'].squeeze(0)
                global_pose = data_dict['global_pose'].squeeze(0)
                cam = data_dict['cam'].squeeze(0)
                jaw = data_dict['jaw'].squeeze(0)
                pose = torch.cat([global_pose, jaw], dim=-1)

                temp, _ = flame(shape)
                vert, _ = flame(shape, exp, pose)

                relative_path = os.path.relpath(file_path, vertex_dir)
                human_id, emotion_id, emotion_level, content_id = relative_path.split(os.sep)[:-1]
                print(human_id, emotion_id, emotion_level, content_id)

                template_dict[(human_id, emotion_id, emotion_level, content_id)] = temp
                vertex_dict[(human_id, emotion_id, emotion_level, content_id)] = vert

    return vertex_dict, template_dict


if __name__ == '__main__':
    audio_dict = load_audio("./trainingdata/audio/")
    vert_dict, temp_dict = load_vertexes("./trainingdata/reconstruction/")

    key_list = []
    audio_list = []
    vert_list = []
    temp_list = []
    for key, value in audio_dict.items():
        if key not in vert_dict or key not in temp_dict:
            print(f"{key} not in vert_dict")
            continue
        key_list.append(key)
        audio_list.append(value)
        vert_list.append(vert_dict[key])
        temp_list.append(temp_dict[key])

    save_npy_mmapped(key_list, "./trainingdata/all_keys.npy")
    save_npy_mmapped(audio_list, "./trainingdata/raw_audio.npy")
    save_pickle(vert_list, "./trainingdata/raw_vertex.pkl")
    save_pickle(temp_list, "./trainingdata/raw_template.pkl")




