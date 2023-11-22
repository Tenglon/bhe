import time
import numpy as np
import torch


def binarize(X, n_bits = 2):

    feat_max, feat_min = torch.max(X), torch.min(X)

    minimum_scale = (feat_max - feat_min) / (2 ** n_bits)
    X_uint8 = torch.floor((X - feat_min) / minimum_scale)
    X_uint8 = X_uint8.clamp(max = 2**n_bits - 1, min = 0).to(torch.uint8)
    
    X_uint8 = X_uint8.clamp(max = 2**n_bits - 1)

    

    nBytes = X.shape[1] // 8 * n_bits

    B = torch.zeros(X.shape[0], nBytes, dtype=torch.uint8).cuda()

    n_groups = 8 // n_bits # each Byte is divided to n_groups

    for i in range(n_groups):
        B_tmp = X_uint8[:, i*nBytes:(i+1)*nBytes] << i * n_bits
        B += B_tmp
        # the bits in B bytes are used as B12|B34|B56|B78 in a 2bits case and B1234|B5678 in a 4bits case.

    return B, X_uint8

def binary_mask(start_bit, n_bits = 2):
    mask = 0
    n_groups = 8 // n_bits # each Byte is divided to n_groups
    for i in range(n_groups):
        mask += (1 << i * n_bits)

    mask = mask << start_bit

    return mask


def bin_count(arr):
    # arr is a 3D tensor, with type uint8
    assert len(arr.shape) == 3
    element_size = 1

    mask = 0xff # 11111111
    s55  = 0x55 # 01010101
    s33  = 0x33 # 00110011
    s0f  = 0x0f # 00001111
    s01  = 0x01 # 00000001

    arr = arr - ((arr >> 1) & s55)
    arr = (arr & s33) + ((arr >> 2) & s33)
    arr = (arr + (arr >> 4)) & s0f

    return (arr * s01) >> (8 * (element_size - 1))



def ext_hamming_dist(B1, B2, n_bits = 2):
    # B1, B2 are binary matrices stored as bytes
    # n_bits is the number of bits per byte

    assert B1.shape[1] == B2.shape[1]
    n_bytes = B1.shape[1] // 8
    # B1_bit_bytes = torch.zeros(B1.shape[0], n_bytes, dtype=torch.uint8).cuda()
    # B2_bit_bytes = torch.zeros(B2.shape[0], n_bytes, dtype=torch.uint8).cuda()

    n_groups = 8 // n_bits # each Byte is divided to n_groups
    # nBytes = B1.shape[1] // n_groups

    dist = torch.zeros(B1.shape[0], B2.shape[0], dtype=torch.int64).cuda()

    for i in range(n_bits):

        mask = binary_mask(i, n_bits)
        B1_bit = B1 & mask
        B2_bit = B2 & mask

        B1_bit_tmp = B1 & 170
        B2_bit_tmp = B2 & 170

        xor_result = B1_bit[:,None,:].bitwise_xor(B2_bit[None,:,:])
        bits_count = bin_count(xor_result)
        bits_count_sum = bits_count.sum(dim=-1)

        dist += bits_count_sum << (2 * i)

    return dist


n_bits = 8

X1 = torch.rand(2, 8).cuda()
X1 = X1 - X1.min() + 1e-8
# X1 = [[1, 0, 0, 0, 0, 0, 0, 0],
#         [1, 1, 0, 0, 0, 0, 0, 0],
#         [0, 1, 1, 0, 0, 0, 0, 0],
#         [1, 0, 1, 1, 0, 0, 0, 0]]
X1 = torch.tensor(X1, dtype=torch.float).cuda()
B1, X1_uint8 = binarize(X1, n_bits)

X2 = torch.rand(5, 8).cuda()
X2 = X2 - X2.min() + 1e-8
# X2 = [[0, 1, 0, 0, 0, 0, 0, 0],
#       [1, 1, 0, 0, 0, 0, 0, 0],
#       [0, 0, 1, 0, 0, 0, 0, 0],
#       [1, 1, 1, 0, 0, 0, 0, 0]]
X2 = torch.tensor(X2, dtype=torch.float).cuda()
B2, X2_uint8 = binarize(X2, n_bits)

t0 = time.time()

dist = ext_hamming_dist(B1, B2, n_bits)
tmp = dist[0,0].item() # call to make sure the computation is done
t1 = time.time()
print(f"hamming_dist time: {t1 - t0}")
euc_dist_sq = (X1[:,None,:]- (X2[None,:,:])).pow(2).sum(dim=-1)
euc_dist_sq = torch.cdist(X1, X2, p=2, compute_mode='use_mm_for_euclid_dist') ** 2

euc_dist_uint8_sq = (X1_uint8[:,None,:] - X2_uint8[None,:,:]).pow(2).sum(dim=-1)
ratio = euc_dist_uint8_sq / euc_dist_sq
# inner_product = X1@X2.T
tmp = euc_dist_sq[0,0].item() # call to make sure the computation is done
t2 = time.time()
print(f"euc_dist_sq time: {t2 - t1}")

import pdb
pdb.set_trace()