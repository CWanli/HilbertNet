import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import spconv.pytorch as spconv
from .utils import Basic_unit

__all__ = ['hilbertnet']


class HilbertNet(nn.Module):
    def __init__(self, num_classes, d1, c2, d2, c3, d3, c4):
        super(HilbertNet, self).__init__()
        self.unit1 = Basic_unit(3, 64, 8, 8, 3, 1, 1, d3, True, c2, d1)
        self.unit2 = Basic_unit(64, 64, 8, 8, 1, 1, 0, d3, True, c3, d2)
        self.unit3 = Basic_unit(64, 128, 8, 16, 1, 1, 0, d3, True, c4, d3)
        self.unit4 = Basic_unit(128, 128, 16, 32, 1, 1, 0, d3, False)
        self.unit5 = Basic_unit(192, 256, 32, 64, 1, 1, 0, d3, False)
        self.unit6 = Basic_unit(320, 512, 64, 128, 1, 1, 0, d3, False)

        self.MLP1 = nn.Sequential(
            nn.Conv1d(in_channels=512+128, out_channels=512, kernel_size=1, stride=1),
            nn.BatchNorm1d(512),
            nn.ReLU(True))
        self.MLP2 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1, stride=1),
            nn.BatchNorm1d(256),
            nn.ReLU(True))
        self.conv1x1 = spconv.SparseSequential(
            spconv.SparseConv2d(1, 8, 1, 1),
            spconv.ToDense(),
            )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.map = nn.AdaptiveMaxPool1d(1)
        self.extract_2d = nn.Sequential(
            nn.Conv2d(128, 128, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 512, 1, 1),
            nn.Sigmoid())
        self.cls_1d = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Linear(64, num_classes),
        )
        self.cls_2d = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Linear(64, num_classes),
        )

    def forward(self, img, xyz):
        B,C,H,W = img.size()
        img = spconv.SparseConvTensor.from_dense(img.permute(0, 2, 3, 1).contiguous())
        in_2d = self.conv1x1(img)
        out1d_1, out2d_1 = self.unit1(in_2d, xyz)
        out1d_2, out2d_2 = self.unit2(out2d_1, out1d_1)
        out1d_3, out2d_3 = self.unit3(out2d_2, out1d_2)
        out1d_4, out2d_4 = self.unit4(out2d_3, out1d_3)
        out1d_5, out2d_5 = self.unit5(out2d_4, torch.cat((out1d_4, out1d_2), 1))
        out1d_6, out2d_6 = self.unit6(out2d_5, torch.cat((out1d_1, out1d_5), 1))
        feat_2d = self.gap(out2d_6).view(B, 128, 1)
        out_1 = self.cls_2d(feat_2d.view(B, 128))
        out_1 = F.log_softmax(out_1, dim=1)
        feat_2d_expand = feat_2d.expand(B, 128, 1024)
        feat_2d = feat_2d.view(B, 128, 1, 1)
        feat_2d_att = self.extract_2d(feat_2d).view(B, 512, 1)
        feat_2d_att = feat_2d_att.expand(B, 512, 1024)
        feat_1d = self.MLP1(torch.cat((out1d_6, feat_2d_expand), 1))
        feat_1d = self.MLP2(feat_1d + feat_2d_att)
        feat_1d = self.map(feat_1d).squeeze(dim=-1)
        out = self.cls_1d(feat_1d)
        out = F.log_softmax(out, dim=1)

        return out, out_1

def hilbertnet(num_classes, d1, c2, d2, c3, d3, c4):
    return HilbertNet(40, d1, c2, d2, c3, d3, c4)