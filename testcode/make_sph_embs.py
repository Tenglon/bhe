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

id2classes = {
    0: 'background',
    1: 'aeroplane',
    2: 'bicycle',
    3: 'bird',
    4: 'boat',
    5: 'bottle',
    6: 'bus',
    7: 'car',
    8: 'cat',
    9: 'chair',
    10: 'cow',
    11: 'dining table',
    12: 'dog',
    13: 'horse',
    14: 'motorbike',
    15: 'person',
    16: 'potted plant',
    17: 'sheep',
    18: 'sofa',
    19: 'train',
    20: 'tv/monitor'
}


if __name__ == '__main__':

    n_cls = len(id2classes)

    # model = Embedding(n_cls, args.dim).cuda()
    embs = nn.Parameter(torch.Tensor(n_cls, args.dim))
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


    embs = embs.detach() / torch.norm(embs.detach(), dim=1, keepdim=True)
    final_obj = similarity(embs)
    print('final obj', final_obj.item())
    outfile = f'sph_emb_{args.dim}d.pth'
    save_dict = {'objects': id2classes, 'embeddings':embs.detach()}
    torch.save(save_dict, outfile)

# test code

    embs = torch.load(f'voc_sph_{args.dim}d.pth')['embeddings']