import torch
from torch import nn
import torch.functional as F

class Block(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.5,
        is_batch_normalization: bool = True
    ):
        super().__init__()

        if is_batch_normalization: 
            self.nor = nn.BatchNorm1d(in_dim)
        else: 
            self.nor = nn.LayerNorm(in_dim) 
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.nor(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x