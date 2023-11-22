import argparse
import torch

from dataset import get_cifar_data, get_imagenet_data, get_mim_data
from pmath import pair_wise_eud, pair_wise_cos, pair_wise_hyp
from metric import Metric
from tqdm import tqdm
import nanopq
import numpy as np

from utils import get_son2parent


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, choices=['cifar100', 'imagenet', 'mim'], default='cifar100', help='dataset name')
parser.add_argument('--top_K', type=int, default=10)
parser.add_argument('--M', type=int, default=8)
parser.add_argument('--n_bits', type=int, default=8)
args = parser.parse_args()

top_K = args.top_K

if args.dataset == "cifar100":
    Xtr, Xte, ytr, yte, hierarchy_csv, label_set, name2clsid = get_cifar_data()
    
elif args.dataset == "imagenet":
    Xtr, Xte, ytr, yte, hierarchy_csv, label_set, name2clsid = get_imagenet_data()

elif args.dataset == "mim":
    Xtr, Xte, ytr, yte, hierarchy_csv, label_set, name2clsid = get_mim_data()

def get_topK_preds(X, y, top_K):

    sim_score = pair_wise_eud(X, X)

    _, indices = torch.sort(sim_score, descending=False, dim=1)

    top_k_preds = y[indices[:, 1:top_K + 1]]

    return top_k_preds


Xtr, Xte = Xtr.cpu().numpy(), Xte.cpu().numpy()
# Instantiate with M=8 sub-spaces
pq = nanopq.PQ(M = args.M, Ks = 2**args.n_bits, verbose=False) 
# M=8 sub-spaces, 256 centroids for each sub-space (8 * 256 = 2048 centroids in total

# Train codewords
pq.fit(Xtr)

# Encode to PQ-codes
Bte = pq.encode(Xte)  # dtype=np.uint8

dist = np.zeros((Xte.shape[0], Xte.shape[0]))

for i, query in tqdm(enumerate(Xte)):
    
    dist[i, :] = pq.dtable(query).adist(Bte)

son2parent = get_son2parent(hierarchy_csv)
metric = Metric(label_set, son2parent)

dist = torch.from_numpy(dist)
_, indices = torch.sort(dist, descending=False, dim=1)
ypred_topk = yte[indices[:, 1:top_K + 1]]

mAP = metric.hop_mAP(ypred_topk, yte, hop = 0)
SmAP = metric.hop_mAP(ypred_topk, yte, hop = 2)
acc = (ypred_topk[:,0] == yte).float().mean().item()

print(f'Dataset: {args.dataset}, M: {args.M}, n_bits: {args.n_bits}, Acc: {acc:.4f}, mAP: {mAP:.4f}, SmAP: {SmAP:.4f}')