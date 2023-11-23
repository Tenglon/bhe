from pathlib import Path
import torch
import argparse

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def similarity(embs):

    norm = torch.norm(embs, dim=1)  # each row is norm 1.

    t1 = norm.unsqueeze(1)                                          # n_cls x 1
    t2 = norm.unsqueeze(0)                                          # 1 x n_cls
    denominator = torch.matmul(t1, t2)                              # n_cls x n_cls, each element is a norm product
    numerator = torch.matmul(embs, embs.t())            # each element is a in-prod
    cos_sim = numerator / denominator                               # n_cls x n_cls, each element is a cos_sim
    cos_sim_off_diag = cos_sim - torch.diag(torch.diag(cos_sim))
    obj = cos_sim_off_diag

    return obj.sum()
    
parser = argparse.ArgumentParser()
parser.add_argument('--dim', type=int, required=True)
parser.add_argument('--lr', type=float, default=1e-4)
args = parser.parse_args()

data_root = '/projects/0/prjs0774/iclr24/data/quickdraw'
files = list(Path(data_root).rglob('*.npy'))
classes = [f.stem for f in files]
classes.sort()
id2classes = {i: c for i, c in enumerate(classes)}


if __name__ == '__main__':

    n_cls = len(id2classes)

    # model = Embedding(n_cls, args.dim).cuda()
    embs = nn.Parameter(torch.Tensor(n_cls, args.dim)).cuda()
    nn.init.normal_(embs, 0, 0.01)

    loss = 0

    for iter in range(1000):

        # normalize embs
        embs = embs.detach() / torch.norm(embs.detach(), dim=1, keepdim=True)
        embs = nn.Parameter(embs)
        optimizer = optim.Adam([embs], args.lr)

        obj = similarity(embs)
        
        optimizer.zero_grad()
        obj.backward()
        optimizer.step()

        loss = obj.item()
        # ema smooth
        loss_ema = loss if iter == 0 else loss_ema * 0.9 + loss * 0.1

        if iter % 10 == 0:
            print('iter', iter, 'loss', loss_ema)


    embs = embs.detach() / (torch.norm(embs.detach(), dim=1, keepdim=True) + 1e-3)
    final_obj = similarity(embs)
    print('final obj', final_obj.item())
    outfile = f'../embs/quick_emb_{args.dim}d.pth'

    save_dict = {'objects': list(id2classes.values()), 'embeddings':embs.detach()}
    torch.save(save_dict, outfile)

# test code

    embs = torch.load(f'../embs/quick_emb_{args.dim}d.pth')['embeddings']