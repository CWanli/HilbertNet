import torch
import torch.nn as nn
import numpy as np
import spconv.pytorch as spconv
from .HilbertPermute import hilbert_permute_3d_to_2d, inv_hilbert_permute_2d_to_3d

class HP(nn.Module):
    def __init__(self, coordinate, distance):
        super(HP, self).__init__()
        self.pool3d = nn.MaxPool3d(kernel_size=2, stride=2)
        self.distance = distance
        self.coordinate = coordinate
    def forward(self, x):
        B,C,H,W = x.size()
        # B,C,W,W,W = x.size()
        x = x.view(B,C,-1,W)
        x_3d = inv_hilbert_permute_2d_to_3d(x, self.distance).view(B,C,W,W,W)
        x_3d = self.pool3d(x_3d)
        x_2d = hilbert_permute_3d_to_2d(x_3d, self.coordinate).view(B,C,-1,W//2)

        return x_2d