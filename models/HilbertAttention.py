import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv

class HA(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(HA, self).__init__()
        inter_channel = in_channel//8 if in_channel//8 > 0 else 1
        self.proj = spconv.SparseSequential(
            spconv.SparseConv3d(in_channel, inter_channel, 1, 1),
            nn.BatchNorm1d(inter_channel),
            nn.LeakyReLU(),
            # spconv.ToDense(),
        )
        self.proj_back = spconv.SparseSequential(
            spconv.SparseConv3d(inter_channel, out_channel, 1, 1),
            nn.BatchNorm1d(out_channel),
            nn.LeakyReLU(),
            spconv.ToDense(),
        )
        self.Q = spconv.SparseSequential(
            spconv.SparseConv3d(inter_channel, inter_channel, (4,1,1), 1, (2,1,0)),
            nn.BatchNorm1d(inter_channel),
            nn.LeakyReLU(),
            spconv.ToDense(),
        )
        self.V = spconv.SparseSequential(
            spconv.SparseConv3d(inter_channel, inter_channel, (1,4,1), 1, (1,2,0)),
            nn.BatchNorm1d(inter_channel),
            nn.LeakyReLU(),
            spconv.ToDense(),
        )
        self.K = spconv.SparseSequential(
            spconv.SparseConv3d(inter_channel, inter_channel, (4,4,1), 1, (2,2,0)),
            nn.BatchNorm1d(inter_channel),
            nn.LeakyReLU(),
            spconv.ToDense(),
        )
        self.feat_proj = spconv.SparseSequential(
            spconv.SparseConv3d(in_channel, out_channel, 1, 1),
            nn.BatchNorm1d(out_channel),
            nn.LeakyReLU(),
            spconv.ToDense(),
        )
        self.Softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor):
        B,C,H,W = x.size()
        x = x.view(B,C,W,W,W)
        x_sp = spconv.SparseConvTensor.from_dense(x.reshape(B,W,W,W,C))
        # create SparseConvTensor manually: see SparseConvTensor.from_dense
        x = self.proj(x_sp)
        query = self.Q(x).view(B,-1,H,W)
        key = self.K(x).view(B,-1,W,H)
        value = self.V(x).view(B,-1,H,W)
        att = torch.einsum('bijk,bikl->bijl', key, query)
        att = self.Softmax(att)
        feat = torch.einsum('bijk,bikl->bijl', value, att)
        B,C,H,W = feat.size()
        feat_sp = spconv.SparseConvTensor.from_dense(feat.reshape(B,W,W,W,C))
        feat_in = self.feat_proj(x_sp).view(B,-1,H,W)
        out = self.proj_back(feat_sp) * feat_in + feat_in

        return out