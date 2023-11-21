import numpy as np
import torch

Xte = torch.rand(1000, 512)
feat_max, feat_min = torch.max(Xte), torch.min(Xte)

n_bits = 2
minimum_scale = (feat_max - feat_min) / (2 ** n_bits)
Xte_uint8 = torch.floor((Xte - feat_min) / minimum_scale).to(torch.uint8)

nBytes = Xte.shape[1] // 8 * n_bits
# nBytes = 128

Bte = torch.zeros(Xte.shape[0], nBytes, dtype=torch.uint8) # each uint8 is 8 bits, used as 4 x 2 bits.
Bte = (Xte_uint8[:nBytes] << 6) + (Xte_uint8[nBytes:2*nBytes] << 4) + (Xte_uint8[2*nBytes:3*nBytes] << 2) + Xte_uint8[3*nBytes:4*nBytes]

import pdb
pdb.set_trace()

n_bits = 4
minimum_scale = (feat_max - feat_min) / (2 ** n_bits)
Xte_uint8 = torch.floor((Xte - feat_min) / minimum_scale).to(torch.uint8)

nBytes = Xte.shape[1] // 8 * n_bits
# nBytes = 256

Bte = torch.zeros(Xte.shape[0], nBytes, dtype=torch.uint8)
Bte = (Xte_uint8[:nBytes] << 4) + Xte_uint8[nBytes:2*nBytes]

import pdb
pdb.set_trace()

