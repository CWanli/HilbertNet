import torch
import numpy as np


def hilbert_permute_3d_to_2d(x, permutation):
    d0, d1, d2, d3, d4 = x.size()
    ret = x[:, :, permutation[0, :], permutation[1, :], :].view(d0, d1, d2, d3, d4)
    return ret


def inv_hilbert_permute_2d_to_3d(x, permutation):
    d0, d1, d2, d3 = x.size()
    ret = x[:, :, permutation, :].view(d0, d1, d2, d3)
    return ret


def hilbert_permute_1d(x, permutation):
    d0, d1, d2 = x.size()
    ret = x[:, :, permutation].view(d0, d1, d2)
    return ret
