import numpy as np
import torch
from torch.utils.data import Dataset


class BdzDataset(Dataset):
    def __init__(self, x, y):
        self.X = torch.tensor(np.array(x), dtype=torch.float32)
        self.y = torch.tensor(np.array(y), dtype=torch.int64)
        self.len = self.y.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        return x, y
