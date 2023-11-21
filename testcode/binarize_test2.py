import numpy as np
import torch


def binarize(X, n_bits = 2):

    feat_max, feat_min = torch.max(X), torch.min(X)

    minimum_scale = (feat_max - feat_min) / (2 ** n_bits)
    X_uint8 = torch.floor((X - feat_min) / minimum_scale).to(torch.uint8)

    nBytes = X.shape[1] // 8 * n_bits

    B = torch.zeros(X.shape[0], nBytes, dtype=torch.uint8)

    n_groups = 8 // n_bits # each Byte is divided to n_groups

    for i in range(n_groups):
        B += X_uint8[:, i*nBytes:(i+1)*nBytes] << i * n_bits

    return B

X = torch.rand(1000, 512)
B = binarize(X, n_bits = 4)

import pdb
pdb.set_trace()