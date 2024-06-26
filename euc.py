from collections import OrderedDict
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.tensorboard as tb
import json
from time import time
from metric import Metric

from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from pmath import pair_wise_eud, pair_wise_cos, pair_wise_hyp
from utils import get_son2parent
from dataset import get_cifar_data, get_imagenet_data, get_mim_data


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

top_K = 10

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--n_bits', type=int, default=4)
parser.add_argument('--emb_dim', type=int, default=128)
parser.add_argument('--hidden_dim', type=int, default=1024)
parser.add_argument('--emb_path', type=str, default='embs/iclr24_cifar_128d.pth')
parser.add_argument('--epochs', type=int, default=200)
# add dataset argument, does not allow default value, must be specified, choices are cifar100 and imagenet
parser.add_argument('--dataset', type=str, choices=['cifar100', 'imagenet', 'mim'], default='cifar100', help='dataset name')
parser.add_argument('--c', type=float, default=0.1, help='curvature for image embeddings')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--workers', type=int, default=0, help='number of workers for dataloader')

args = parser.parse_args()
r = np.sqrt(1/args.c)
writer = tb.SummaryWriter(log_dir=f'runs/{args.dataset}'.format(args.dataset))

if args.dataset == "cifar100":
    Xtr, Xte, ytr, yte, hierarchy_csv, label_set, name2clsid = get_cifar_data()
    
elif args.dataset == "imagenet":
    Xtr, Xte, ytr, yte, hierarchy_csv, label_set, name2clsid = get_imagenet_data()

elif args.dataset == "mim":
    Xtr, Xte, ytr, yte, hierarchy_csv, label_set, name2clsid = get_mim_data()


# embs = torch.rand((len(label_set), 128))
son2parent = get_son2parent(hierarchy_csv)
# emb_data = torch.load(args.emb_path)
# embs_preorder = emb_data['embeddings']
# names_preorder = emb_data['objects']

embs = torch.rand((len(label_set), args.emb_dim))
embs_cuda = embs.cuda()

# for i, name in enumerate(names_preorder):
#     if name in name2clsid:
#         embs[name2clsid[name], :] = embs_preorder[i]
# embs_cuda = copy.deepcopy(embs).cuda()

if torch.sum(embs == 0) > 0:
    raise ValueError("Some classes are missing in the embedding file.")

Xtr, ytr, Xte, yte = Xtr.cpu(), ytr.cpu(), Xte.cpu(), yte.cpu()
train_dataset = FeatureDataset(Xtr, ytr)
test_dataset = FeatureDataset(Xte, yte)

train_dataloader = DataLoader(train_dataset, batch_size=2048, shuffle=True, num_workers=args.workers)
test_dataloader = DataLoader(test_dataset, batch_size=2048, shuffle=False, num_workers=args.workers)

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
        # if (x.norm(dim=1) >= r).any():
            # x = r * x / (x.norm(dim=1,keepdim=True) + 1e-2)

        return x

    
def get_topK_preds(X, y, top_K):

    sim_score = pair_wise_cos(X, X)

    _, indices = torch.sort(sim_score, descending=False, dim=1)

    top_k_preds = y[indices[:, 1:top_K + 1]]

    return top_k_preds


model = MLP(Xtr.shape[1], args.hidden_dim, embs.shape[1])
model = model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)
metric = Metric(label_set, son2parent)

re_calculate_count = 0

for epoch in range(args.epochs):

    model.train()
    tr_losses = []

    st = time()
    
    for i, (X, y, A) in enumerate(train_dataloader):

        X, y, A = X.cuda(), y.cuda(), A.cuda()

        Apred = model(X)


        loss = loss_fn(y, Apred, pair_wise_eud, c = args.c, T = 1)

        while loss.isnan().any():
            loss = loss_fn(y, Apred, pair_wise_eud, c = args.c, T = 1)
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

                X, y, A = X.cuda(), y.cuda(), A.cuda()

                Apred = model(X)
                Apred_list.append(Apred.detach().cpu())

                loss = loss_fn(y, Apred, pair_wise_eud, c = args.c, T = 1)

                while loss.isnan().any():
                    loss = loss_fn(y, Apred, pair_wise_eud, c = args.c, T = 1)
                    re_calculate_count += 1
                    
                te_losses.append(loss.detach().cpu().item())

            Apred = torch.cat(Apred_list, axis = 0)

        st = time()
        ypred_topk = get_topK_preds(Apred, yte, top_K)
        mAP = metric.hop_mAP(ypred_topk, yte, hop = 0)
        SmAP = metric.hop_mAP(ypred_topk, yte, hop = 2)
        acc = (ypred_topk[:,0] == yte).float().mean().item()
        eval_time = time() - st

        print(f'Epoch: {epoch}, Train Loss: {np.mean(tr_losses):.4f}, Test Loss: \
            {np.mean(te_losses):.4f}, Epoch Time: {epoch_time:.4f}, Eval Time: \
            {eval_time:.4f}, Acc: {acc:.4f}, mAP: {mAP:.4f}, SmAP: {SmAP:.4f}')

        # log on tensorboard
        writer.add_scalar('train_loss', np.mean(tr_losses), epoch)
        writer.add_scalar('test_loss', np.mean(te_losses), epoch)
        writer.add_scalar('acc', acc, epoch)
        writer.add_scalar('mAP', mAP, epoch)
        writer.add_scalar('SmAP', SmAP, epoch)

        # add exponential moving average to acc and mAP on tensorboard
        if epoch == 0:
            ema_acc = acc
            ema_mAP = mAP
            ema_SmAP = SmAP
        else:
            ema_acc = 0.9 * ema_acc + 0.1 * acc
            ema_mAP = 0.9 * ema_mAP + 0.1 * mAP
            ema_SmAP = 0.9 * ema_SmAP + 0.1 * SmAP

        writer.add_scalar('ema_acc', ema_acc, epoch)
        writer.add_scalar('ema_mAP', ema_mAP, epoch)
        writer.add_scalar('ema_SmAP', ema_SmAP, epoch)

    # save model per 10 epoch since epoch 20
    if epoch >= 20 and epoch % 50 == 0:
        emb_dim_str = args.emb_path.split("/")[-1].split(".")[0].split("_")[-1]
        torch.save(model.state_dict(), f'runs/{args.dataset}_model_eucfloat_{emb_dim_str}_{epoch}.pth')