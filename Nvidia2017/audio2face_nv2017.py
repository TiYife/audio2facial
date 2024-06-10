import numpy as np
import torch
import torch.nn as nn


class Audio2Mesh(nn.Module):
    """https://research.nvidia.com/sites/default/files/publications/karras2017siggraph-paper_0.pdf"""

    def __init__(self,
                 n_verts: int,
                 n_onehot: int,
                 n_emotion_dim: int,
                 n_max_frame: int,
                 n_max_sentences: int
                 ):
        super().__init__()
        self.n_verts = n_verts
        self.n_onehot = n_onehot

        self.emotion_set = None
        self.emotion_dim = n_emotion_dim
        emotion_set = np.random.normal(0.0, 1.0, (n_max_sentences, n_max_frame, n_emotion_dim))
        self.emotion_set = nn.Parameter(torch.from_numpy(emotion_set).float(), requires_grad=True)

        self.analysis_net = nn.Sequential(
            nn.Conv2d(1, 72, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(72),
            nn.ReLU(),
            nn.Conv2d(72, 108, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(108),
            nn.ReLU(),
            nn.Conv2d(108, 162, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(162),
            nn.ReLU(),
            nn.Conv2d(162, 243, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(243),
            nn.ReLU(),
            nn.Conv2d(243, 256, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.articulation_net = nn.Sequential(
            nn.Conv2d(256 + self.emotion_dim, 256 + self.emotion_dim, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(256 + self.emotion_dim),
            nn.ReLU(),
            nn.Conv2d(256 + self.emotion_dim, 256 + self.emotion_dim, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(256 + self.emotion_dim),
            nn.ReLU(),
            nn.Conv2d(256 + self.emotion_dim, 256 + self.emotion_dim, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(256 + self.emotion_dim),
            nn.ReLU(),
            nn.BatchNorm2d(256 + self.emotion_dim),
            nn.Conv2d(256 + self.emotion_dim, 256 + self.emotion_dim, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.BatchNorm2d(256 + self.emotion_dim),
            nn.Conv2d(256 + self.emotion_dim, 256 + self.emotion_dim, kernel_size=(4, 1), stride=(4, 1)),
            nn.ReLU(),
        )

        self.output_net = nn.Sequential(
            nn.Linear(256 + n_onehot + self.emotion_dim, 72),
            nn.Linear(72, 128),
            nn.Tanh(),
            nn.Linear(128, 50),
            nn.Linear(50, n_verts),
        )

    def forward(self, x, one_hot, template, emo_data, sentence_id, emo_idx, **kwargs):
        bs = x.size(0)
        onehot_embedding = one_hot.repeat(1, 32).view(bs, 1, -1, 32)
        # 128x52x32
        x = x.unsqueeze(1)
        # 128x1x52x32
        # x = self.instance_norm(x)
        x = self.analysis_net(torch.cat((x, onehot_embedding), 2))

        if emo_data is not None:
            # emotion = emo_data[emo_idx.chunk(chunks=1, dim=0)]
            emotion = emo_data
            e = emotion.view(emotion.view(-1, self.emotion_dim).size()[0], self.emotion_dim, 1, 1)
            e = e.expand(x.size()[0], self.emotion_dim, 64, 1)
            x = torch.cat((x, e), dim=1)
            x = x.view(-1, 256 + self.emotion_dim, 64, 1)
        else:
            # idx = torch.stack((sentence_id, emo_idx), dim=1).chunk(chunks=1, dim=0)
            emotion = self.emotion_set[sentence_id, emo_idx]
            e = emotion.view(x.size()[0], self.emotion_dim, 1, 1)
            e = e.expand(x.size()[0], self.emotion_dim, 64, 1)
            x = torch.cat((x, e), dim=1)
            x = x.view(-1, 256 + self.emotion_dim, 64, 1)

        # 128x256x64x1
        x = self.articulation_net(x)
        # 128x256x1x1
        x = x.view(x.size(0), -1)
        # 128x256
        x = self.output_net(torch.cat((x, one_hot), 1))
        # 128x15069
        return x.view(bs, -1, 3) + template, emotion.view(-1, self.emotion_dim)

    def predict(self, x, one_hot, template, **kwargs):
        return self(x, one_hot, template, None, None, **kwargs)

