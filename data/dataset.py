"""
Dataset Classes for Vessel Trajectory Data
=========================================

PyTorch Dataset implementations for trajectory data handling.
"""

import torch
from torch.utils.data import Dataset
from typing import Tuple


class TrajectoryDataset(Dataset):
    """Dataset class for trajectory data"""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]