from tqdm import tqdm
import glob, os
import numpy as np
from pathlib import Path

data_root = '/projects/0/prjs0774/iclr24/data/quickdraw'

files = list(Path(data_root).rglob('*.npy'))

X_list = []
label_list = []

for file in tqdm(files):
    
    X = np.load(file)
    label = file.stem
    X_list.append(X)
    label_list.extend([label] * X.shape[0])

data = np.concatenate(X_list, axis=0)

np.savez('quickdraw_full.npz', data=data, label=label_list)
data = np.load(os.path.join('quickdraw_full.npz'))

X, y = data['data'], data['label']


