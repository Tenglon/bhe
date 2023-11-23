from collections import OrderedDict
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import json
from time import time
from bin_utils import binarize, ext_hamming_dist
from metric import Metric

from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from pmath import pair_wise_eud, pair_wise_cos, pair_wise_hyp
from sklearn.model_selection import train_test_split

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--n_bits', type=int, default=4)
parser.add_argument('--emb_dim', type=int, default=128)
parser.add_argument('--hidden_dim', type=int, default=1024)
parser.add_argument('--epochs', type=int, default=200)
# add dataset argument, does not allow default value, must be specified, choices are cifar100 and imagenet
parser.add_argument('--c', type=float, default=0.1, help='curvature for image embeddings')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--workers', type=int, default=0, help='number of workers for dataloader')
args = parser.parse_args()
r = np.sqrt(1/args.c)

def loss_fn(y, Apred, dist_func, c, T):

    logits = -dist_func(Apred, embs_cuda ,c) / T
    loss = F.cross_entropy(logits, y)
    
    return loss

# create dataset and dataloader
class FeatureDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __getitem__(self, index):

        feat = self.X[index]
        label = self.y[index]
        emb = embs[label]

        return feat, label, emb
    
    def __len__(self):
        return len(self.X)
    

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.relu = nn.LeakyReLU()

        self.fc1 = nn.Linear(input_dim, 1024)
        # self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, hidden_dim)
        # self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        # x = self.bn1(x)
        x=  self.relu(self.fc2(x))
        # x = self.bn2(x)
        x = self.fc3(x)
        if (x.norm(dim=1) >= r).any():
            x = r * x / (x.norm(dim=1,keepdim=True) + 1e-2)

        return x
    

def ext_hamming_dist_blockwise(B1, B2, n_bits):

    block_size = 4096

    dist = torch.zeros(B1.shape[0], B2.shape[0]).cuda()
    B1chunks = torch.split(B1, block_size, dim=0)
    B2chunks = torch.split(B2, block_size, dim=0)
    for i, Bchunk1 in enumerate(B1chunks):
        for j, Bchunk2 in enumerate(B2chunks):
            dist[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = ext_hamming_dist(Bchunk1, Bchunk2, args.n_bits)

    return dist


def get_topK_preds_bin(X, y, top_K):
    
    B, _ = binarize(X, args.n_bits)
    B = B.cuda()

    dist = ext_hamming_dist_blockwise(B, B, args.n_bits)

    if dist.shape[0] > 30000:
        dist = dist.cpu()

    _, indices = torch.sort(dist, descending=False, dim=1)
    indices = indices.cpu()

    top_k_preds = y[indices[:, 1:top_K + 1]]

    return top_k_preds


emb_file = f'./embs/quick_emb_{args.emb_dim}d.pth'
emb_data = torch.load(emb_file)

label_set = emb_data['objects']
label_set.sort()

name2clsid = {v: k for k, v in enumerate(label_set)}


embs = torch.rand((len(label_set), args.emb_dim))
embs_preorder = emb_data['embeddings']
names_preorder = emb_data['objects']

for i, name in enumerate(names_preorder):
    if name in name2clsid:
        embs[name2clsid[name], :] = embs_preorder[i]
embs_cuda = copy.deepcopy(embs).cuda()

data = np.load('feat/quickdraw_full.npz')
X, y = data['data'], data['label']

y = np.array([label_set.index(label) for label in y])

son2parent = OrderedDict({label: 'Root' for label in label_set})

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.99, random_state=42)
yte = torch.from_numpy(yte)

Xte_small, yte_small = Xte[:50000], yte[:50000]

train_dataset = FeatureDataset(Xtr, ytr)
test_dataset = FeatureDataset(Xte_small, yte_small)

train_dataloader = DataLoader(train_dataset, batch_size=2048, shuffle=True, num_workers=args.workers)
test_dataloader = DataLoader(test_dataset, batch_size=2048, shuffle=False, num_workers=args.workers)

model = MLP(Xtr.shape[1], args.hidden_dim, embs.shape[1])
model = model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)
metric = Metric(label_set, son2parent)

top_K = 10

re_calculate_count = 0
best_mAP = 0

print(f"n_bits: {args.n_bits}, emb_dim: {args.emb_dim}")

for epoch in range(args.epochs):

    model.train()
    tr_losses = []

    st = time()
    
    for i, (X, y, A) in enumerate(train_dataloader):

        X, y, A = X.float().cuda(), y.cuda(), A.cuda()

        Apred = model(X)

        loss = loss_fn(y, Apred, pair_wise_hyp, c = args.c, T = 1)

        while loss.isnan().any():
            loss = loss_fn(y, Apred, pair_wise_hyp, c = args.c, T = 1)
            re_calculate_count += 1

            if re_calculate_count % 100 == 0:
                print(f"re_calculate_count: {re_calculate_count}")

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        tr_losses.append(loss.item())
    epoch_time = time() - st

    # Get new feature
    if epoch % 10 == 0:
        with torch.no_grad():

            Apred_list = []
            te_losses = []

            for i, (X, y, A) in enumerate(test_dataloader):

                X, y, A = X.float().cuda(), y.cuda(), A.cuda()

                Apred = model(X)
                Apred_list.append(Apred.detach().cpu())

                loss = loss_fn(y, Apred, pair_wise_hyp, c = args.c, T = 1)

                while loss.isnan().any():
                    loss = loss_fn(y, Apred, pair_wise_hyp, c = args.c, T = 1)
                    re_calculate_count += 1
                    
                te_losses.append(loss.detach().cpu().item())

            Apred = torch.cat(Apred_list, axis = 0)

        st = time()
        ypred_topk = get_topK_preds_bin(Apred, yte_small, top_K)
        mAP = metric.hop_mAP(ypred_topk, yte_small, hop = 0)
        SmAP = metric.hop_mAP(ypred_topk, yte_small, hop = 2)
        acc = (ypred_topk[:,0] == yte_small).float().mean().item()
        eval_time = time() - st

        print(f'Acc: {acc:.4f}, epoch_time: {epoch_time:.4f}, eval_time: {eval_time:.4f}, mAP: {mAP:.4f}, SmAP: {SmAP:.4f}')

        if mAP > best_mAP:
            best_mAP = mAP
            best_model = copy.deepcopy(model)
            torch.save(best_model.state_dict(), f'./checkpoints/quick_hyp_{args.n_bits}bits_{args.emb_dim}dim.pth')
            print(f'Best mAP: {best_mAP:.4f}, model saved!')