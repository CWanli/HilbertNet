import torch
import torch.nn as nn
import torch.nn.functional as F
from .HilbertPermute import hilbert_permute_1d

class HI(nn.Module):
    def __init__(self, channel, point, distance):
        super(HI, self).__init__()
        self.channel = channel
        self.distance = distance
        self.point = point
        self.avg_filter = nn.AvgPool2d(2,2)

    def forward(self, img):
        B,C,H,W = img.size()
        weight = img
        weight = (weight[:,:,0:H,0:W] > 1e-6).float()
        weight = 1 - self.avg_filter(weight) + 1/4
        weight = F.interpolate(weight, size=[H,W], mode='nearest')
        img = img*weight
        img = img.view(B,self.channel,-1,1)
        point = F.interpolate(img, size=[self.point, 1], mode='bilinear',align_corners=True)
        point = point.view(B,self.channel,-1)
        point = hilbert_permute_1d(point, self.distance)

        return point