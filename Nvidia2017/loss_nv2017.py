import torch
from loss import VocaLoss


class VocaEmotionLoss(VocaLoss):
    def __init__(self, k_rec: float = 1.0, k_vel: float = 10.0, k_emo: float = 1.0):
        super().__init__(k_rec, k_vel)
        self.k_emo = k_emo

    def emotion_loss(self, pred_e):
        if pred_e is None:
            return torch.tensor(0)

        n_consecutive_frames = 2
        pred = pred_e.view((-1, n_consecutive_frames) + pred_e.size()[1:])

        return torch.mean(torch.sum((pred[:, 1] - pred[:, 0]) ** 2, axis=1))

    def __call__(self, pred, gt, e=None, **kwargs):
        bs = pred.shape[0]
        gt = gt.view(bs, -1, 3)
        pred = pred.view(bs, -1, 3)
        self.n_verts = pred.shape[1]

        rec_loss = self.reconstruction_loss(pred, gt)
        vel_loss = self.velocity_loss(pred, gt)
        emo_loss = self.emotion_loss(e)

        return {
            "loss": rec_loss * self.k_rec + vel_loss * self.k_vel + emo_loss * self.k_emo,
            "rec_loss": rec_loss,
            "vel_loss": vel_loss,
            "emo_loss": emo_loss,
        }
