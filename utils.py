from collections import OrderedDict
import queue
import numpy as np

import pandas as pd
import torch
from torch.distributions import Categorical
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


class Node:

    def __init__(self, name, son2parent):

        self.name = name
        self.son2parent = son2parent
        self.children = self.construct_children()

        self.n_children = len(self.children) if self.children is not None else 0
        self.n_descendants = self.get_n_descendants()

        # if self.n_descendants > 0:
            # print(self.name, 'has', self.n_descendants, 'descendants')
            
    # Construct the children recursively
    def construct_children(self):
        if self.name not in self.son2parent.values():
            return None
        else:
            children = []
            for son, parent in self.son2parent.items():
                if parent == self.name:
                    children.append(Node(son, self.son2parent))
            return children
        
    def get_n_descendants(self):
        if self.n_children == 0:
            return 0
        else:
            return self.n_children + sum([child.get_n_descendants() for child in self.children])
        
    def __str__(self) -> str:
        return self.name + ', n_children: ' + str(self.n_children) + ', n_descendants: ' + str(self.n_descendants)

def get_son2parent(csv_path):

    son2parent = OrderedDict()
    with open (csv_path,'r') as fp:
        lines = fp.readlines()
        for line in lines:
            if line[-1] == '\n':
                line = line[:-1]
            tmp_list = line.split(',')
            for i in range(len(tmp_list) - 2): # 最后一位是数字，不管
                key, value = tmp_list[i], tmp_list[i+1]
                son2parent[key] = value
    if '' in son2parent.keys():
        del son2parent['']
        
    return son2parent

def load_edge_list(son2parent):

    df = pd.DataFrame(list(son2parent.items()), columns=['son', 'parent'])
    df.dropna(inplace=True)

    reshape_links = df[['son', 'parent']].values.reshape(-1)# from n_cls x 2 to (2n_cls,)
    idx, objects = pd.factorize(reshape_links) # 拆分成idx和字典两部分，idx存的是字典中的坐标。
    idx = idx.reshape(-1, 2).astype('int')
    weights = np.ones(idx.shape[0])
    
    return idx, objects.tolist(), weights

def get_sample_weights(root):
    
        sample_weights = OrderedDict()
        sample_weights[root.name] = root.n_descendants

        q = queue.Queue()
        q.put(root)
        while not q.empty():
            node = q.get()
            if node.children is not None:
                for child in node.children:
                    sample_weights[child.name] = child.n_descendants if child.n_descendants > 0 else 1
                    q.put(child)

        return sample_weights


def read_fbin(filename, start_idx=0, chunk_size=None):
    """ Read *.fbin file that contains float32 vectors
    Args:
        :param filename (str): path to *.fbin file
        :param start_idx (int): start reading vectors from this index
        :param chunk_size (int): number of vectors to read. 
                                 If None, read all vectors
    Returns:
        Array of float32 vectors (numpy.ndarray)
    """
    with open(filename, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
        nvecs = (nvecs - start_idx) if chunk_size is None else chunk_size
        arr = np.fromfile(f, count=nvecs * dim, dtype=np.float32, 
                          offset=start_idx * 4 * dim)
    return arr.reshape(nvecs, dim)


def read_ibin(filename, start_idx=0, chunk_size=None):
    """ Read *.ibin file that contains int32 vectors
    Args:
        :param filename (str): path to *.ibin file
        :param start_idx (int): start reading vectors from this index
        :param chunk_size (int): number of vectors to read.
                                 If None, read all vectors
    Returns:
        Array of int32 vectors (numpy.ndarray)
    """
    with open(filename, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
        nvecs = (nvecs - start_idx) if chunk_size is None else chunk_size
        arr = np.fromfile(f, count=nvecs * dim, dtype=np.int32, 
                          offset=start_idx * 4 * dim)
    return arr.reshape(nvecs, dim)

class GraphDataset(Dataset):

    def __init__(self, son2parent, opt):

        self.opt = opt

        self.edges, self.objects, _ = load_edge_list(son2parent)

        root = Node('Root', son2parent)
        sample_weights = get_sample_weights(root)

        # normalize the sample_weights's values as probabilities
        probs = torch.tensor(list(sample_weights.values()))
        probs = probs / probs.sum()


        # create a categorical distribution over the probs
        self.distrib = Categorical(probs)

    def __len__(self):

        return len(self.objects)
    
    def __getitem__(self, idx):
            
        return self.get_batch()

    def get_batch(self):

        unif = torch.ones(self.edges.shape[0])
        idx = unif.multinomial(self.opt.batchsize, replacement=True)

        batch = torch.zeros(self.opt.batchsize, self.opt.nnegs + 2, dtype=torch.long)

        tail, head = self.edges[idx, 0], self.edges[idx, 1]
        batch[:, 0], batch[:, 1] = torch.from_numpy(tail), torch.from_numpy(head)

        negs = self.distrib.sample((self.opt.batchsize, self.opt.nnegs))

        tail_bad, head_bad = np.equal(negs, tail[:, None]), np.equal(negs, head[:, None])
        tail_bad_x, tail_bad_y = np.where(tail_bad)
        head_bad_x, head_bad_y = np.where(head_bad)

        while tail_bad.any() or head_bad.any():

            # print('bad sample number:', len(tail_bad_x) + len(head_bad_x))

            if tail_bad.any():
                negs[tail_bad_x, tail_bad_y] = self.distrib.sample((len(tail_bad_x),))

            if head_bad.any():
                negs[head_bad_x, head_bad_y] = self.distrib.sample((len(head_bad_x),))

            tail_bad = np.equal(negs, tail[:, None])
            tail_bad_x, tail_bad_y = np.where(tail_bad)
            head_bad = np.equal(negs, head[:, None])
            head_bad_x, head_bad_y = np.where(head_bad)

        batch[:, 2:] = negs

        # targets are always zero, because the first two nodes are always connected
        # see forward() in Embedding class.
        targets = torch.zeros(self.opt.batchsize, dtype=torch.long) 

        return batch, targets
    
class Options:
    def __init__(self):
        # self.dim = 300
        # self.c, self.T = 0.1, 1
        # self.manifold = PoincareBallExact(c=self.c)
        # self.sparse = False
        # self.lr = 0.1
        self.batchsize = 10
        self.nnegs = 50
        # self.epochs = 4000
        # self.burnin = 20
        # self.dampening = 0.75
        # self.ndproc = 1
        
opt = Options()

if __name__ == '__main__':

    # tree_file = './activity_net_depth_v5.csv'
    tree_file = './kinetics_depth_v2.csv'
    # tree_file = './moments_depth_v4.csv'

    son2parent = get_son2parent(tree_file)

    dataset = GraphDataset(son2parent, opt)

    batch = dataset.get_batch()

    