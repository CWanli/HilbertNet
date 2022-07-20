import torch
import torch.nn as nn
from .HilbertAttention import HA
from .HilbertPool import HP
from .HilbertInterp import HI

class Basic_unit(nn.Module):
    def __init__(self, in_channel1d, out_channel1d, in_channel2d, out_channel2d, k, s, p, dist, is_pool, coord=None, distance=None):
        super(Basic_unit, self).__init__()
        self.S_MLP = nn.Sequential(
            nn.Conv1d(in_channels=in_channel1d, out_channels=out_channel1d, kernel_size=k, stride=s, padding=p),
            nn.BatchNorm1d(out_channel1d),
            nn.ReLU(True)
        )
        self.HI = HI(out_channel1d, 1024, dist)
        self.HA = HA(in_channel2d, out_channel2d)
        self.is_pool = is_pool
        if is_pool:
            self.HP = HP(coord, distance)

    def forward(self, img, xyz):
        if self.is_pool:
            out2d = self.HP(self.HA(img))
        else:
            out2d = self.HA(img)
        out1d = self.S_MLP(xyz) + self.HI(out2d)

        return out1d, out2d