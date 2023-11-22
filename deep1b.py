from time import time
from utils import read_fbin, read_ibin
import os

if __name__ == '__main__':

    deep1b_root = '/projects/0/prjs0774/deep1b'

    fbin_1m = 'base.1M.fbin'
    fbin_10m = 'base.10M.fbin'
    fbin_350m = 'learn.350M.fbin'
    fbin_1b = 'base.1B.fbin'

    fbin_1m = os.path.join(deep1b_root, fbin_1m)
    fbin_10m = os.path.join(deep1b_root, fbin_10m)
    fbin_350m = os.path.join(deep1b_root, fbin_350m)
    fbin_1b = os.path.join(deep1b_root, fbin_1b)

    ibin_gt100k  = 'groundtruth.public.100K.ibin'
    ibin_gt10k   = 'groundtruth.public.10K.ibin'
    fbin_qry100k = 'query.public.100K.fbin'
    fbin_qry10k  = 'query.public.10K.fbin'

    # t0 = time()
    # feat_1m = read_fbin(fbin_1m)
    # t1 = time()
    # print('Read 1M vectors in {} seconds'.format(t1 - t0))

    # t0 = time()
    # feat_10m = read_fbin(fbin_10m)
    # t1 = time()
    # print('Read 10M vectors in {} seconds'.format(t1 - t0))
    
    # t0 = time()
    # feat_350m = read_fbin(fbin_350m)
    # t1 = time()
    # print('Read 350M vectors in {} seconds'.format(t1 - t0))

    # t0 = time()
    # feat_1b = read_fbin(fbin_1b)
    # t1 = time()
    # print('Read 1B vectors in {} seconds'.format(t1 - t0))
    