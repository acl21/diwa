import numpy as np
import torch
from torch.utils.data import Dataset


# Custom Dataset
class RewClsDataset(Dataset):
    def __init__(self, file_path: str):
        """
        Data has "X", "y" keys.
        """
        data = np.load(file_path, allow_pickle=True)
        self.X = torch.tensor(data["X"], dtype=torch.float32)
        self.y = torch.tensor(data["y"], dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
