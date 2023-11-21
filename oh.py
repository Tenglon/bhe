from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import json
from time import time
from metric import Metric

from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from pmath import pair_wise_eud, pair_wise_cos, pair_wise_hyp
from utils import get_son2parent
from dataset import get_cifar_data, get_imagenet_data, get_mim_data


def loss_fn(support, query, dist_func, c, T):
    #Here we use synthesised support.
    logits = -dist_func(support,query,c) / T
    fewshot_label = torch.arange(support.size(0)).cuda()
    loss = F.cross_entropy(logits, fewshot_label)
    
    return loss

# create dataset and dataloader
class FeatureDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __getitem__(self, index):
        feat = self.X[index]
        label = self.y[index]
        return feat, label
    
    def __len__(self):
        return len(self.X)

top_K = 10

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--hidden_dim', type=int, default=1024)
parser.add_argument('--epochs', type=int, required=True)
# add dataset argument, does not allow default value, must be specified, choices are cifar100 and imagenet
parser.add_argument('--dataset', type=str, choices=['cifar100', 'imagenet', 'mim'], required=True)
args = parser.parse_args()

if args.dataset == "cifar100":
    Xtr, Xte, ytr, yte, hierarchy_csv, label_set = get_cifar_data()
    
elif args.dataset == "imagenet":
    Xtr, Xte, ytr, yte, hierarchy_csv, label_set = get_imagenet_data()

elif args.dataset == "mim":
    Xtr, Xte, ytr, yte, hierarchy_csv, label_set = get_mim_data()

son2parent = get_son2parent(hierarchy_csv)

train_dataset = FeatureDataset(Xtr, ytr)
test_dataset = FeatureDataset(Xte, yte)
train_dataloader = DataLoader(train_dataset, batch_size=2048, shuffle=True, num_workers=0)
test_dataloader = DataLoader(test_dataset, batch_size=2048, shuffle=False, num_workers=0)


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(input_dim, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.bn1(x)
        # x = x / x.norm(dim = 1).view(-1, 1) # re-normalize
        x = self.softmax(self.fc2(x))

        return x

    
def get_topK_preds(X, y, top_K):

    sim_score = pair_wise_cos(X.float(), X.float())

    _, indices = torch.sort(sim_score, descending=False, dim=1)

    top_k_preds = y[indices[:, 1:top_K + 1]]

    return top_k_preds

model = MLP(Xtr.shape[1], ytr.unique().nelement())
model = model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
metric = Metric(label_set, son2parent)

for epoch in range(100):

    model.train()
    tr_losses = []
    st = time()
    for i, (X, y) in enumerate(train_dataloader):
        X, y = X.cuda(), y.cuda()
        y_pred = model(X)
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tr_losses.append(loss.item())

    # Get new feature
    with torch.no_grad():

        feat_extractor = torch.nn.Sequential(*list(model.children())[:-2], nn.LeakyReLU())

        new_Xte_list = []
        acc_list = []
        te_losses = []

        for i, (X, y) in enumerate(test_dataloader):

            X, y = X.cuda(), y.cuda()

            new_Xte_batch = feat_extractor(X)
            # new_Xte_batch = new_Xte_batch / new_Xte_batch.norm(dim = 1).view(-1, 1)

            new_Xte_list.append(new_Xte_batch.detach().cpu())

            y_pred = model(X)
            loss = criterion(y_pred, y).item()
            acc_batch = (y_pred.argmax(dim=1) == y).sum().item() / len(y)
            acc_list.append(acc_batch)
            te_losses.append(loss)

        new_Xte = torch.cat(new_Xte_list, axis = 0)

    epoch_time = time() - st

    st = time()
    ypred_topk = get_topK_preds(new_Xte, yte, top_K)
    mAP = metric.hop_mAP(ypred_topk, yte, hop = 0)
    SmAP = metric.hop_mAP(ypred_topk, yte, hop = 2)
    acc = (ypred_topk[:,0] == yte).float().mean().item()
    eval_time = time() - st

    print(f'Epoch: {epoch}, Train Loss: {np.mean(tr_losses):.4f}, Test Loss: \
          {np.mean(te_losses):.4f}, Epoch Time: {epoch_time:.4f}, Eval Time: \
          {eval_time:.4f}, Acc: {acc:.4f}, mAP: {mAP:.4f}, SmAP: {SmAP:.4f}')