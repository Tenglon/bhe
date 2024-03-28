from time import time
import torch
from torch.utils.data import DataLoader, Dataset
from pmath import pair_wise_eud, pair_wise_cos, pair_wise_hyp
from bin_utils import binarize, ext_hamming_dist
from dataset import get_cifar_data, get_imagenet_data, get_mim_data

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, choices=['cifar100', 'imagenet', 'mim'], default='imagenet', help='dataset name')
args = parser.parse_args()

if args.dataset == "cifar100":
    Xtr, Xte, ytr, yte, hierarchy_csv, label_set, name2clsid = get_cifar_data()
    
elif args.dataset == "imagenet":
    Xtr, Xte, ytr, yte, hierarchy_csv, label_set, name2clsid = get_imagenet_data()

elif args.dataset == "mim":
    Xtr, Xte, ytr, yte, hierarchy_csv, label_set, name2clsid = get_mim_data()


def ext_hamming_dist_blockwise(B1, B2, n_bits):

    dist = torch.zeros(B1.shape[0], B2.shape[0])
    if B1.is_cuda:
        dist = dist.cuda()
    B1chunks = torch.split(B1, block_size, dim=0)
    B2chunks = torch.split(B2, block_size, dim=0)
    for i, Bchunk1 in enumerate(B1chunks):
        for j, Bchunk2 in enumerate(B2chunks):
            dist[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = ext_hamming_dist(Bchunk1, Bchunk2, n_bits)

    return dist

query_size = 100
db_size = 10 ** 5
use_cuda = True

t00 = time()

F1 = torch.rand((query_size, 512))
F2 = torch.rand((db_size, 512))
if use_cuda:
    F1, F2 = F1.cuda(), F2.cuda()

print(f'init time: {time() - t00}')

t0 = time()
# dist = pair_wise_eud(F1, F2)
block_size = 4096
dist = torch.zeros(F1.shape[0], F2.shape[0])
if F1.is_cuda:
    dist = dist.cuda()
F1chunks = torch.split(F1, block_size, dim=0)
F2chunks = torch.split(F2, block_size, dim=0)
for i, Fchunk1 in enumerate(F1chunks):
    for j, Fchunk2 in enumerate(F2chunks):
        dist[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = torch.cdist(Fchunk1, Fchunk2, compute_mode='use_mm_for_euclid_dist')
        torch.cuda.synchronize()

print(f'Euclidean distance: {time() - t0}')

n_bits = 2
for embed_dim in [4, 8, 16, 32, 64, 128, 256]:
    nbytes = embed_dim * n_bits // 8

    B1 = torch.randint(low=0, high=256, size=(query_size, nbytes), dtype=torch.uint8)
    B2 = torch.randint(low=0, high=256, size=(db_size, nbytes), dtype=torch.uint8)

    if F1.is_cuda:
        B1, B2 = B1.cuda(), B2.cuda()

    t0 = time()
    tmp = ext_hamming_dist_blockwise(B1, B2, n_bits)
    # torch.cuda.synchronize()
    t1 = time()
    t = t1 - t0

    print(f'bits: {n_bits}, dim: {embed_dim}, n_bits: {nbytes*8}', f'Hamming distance time: {t}')