import torch
from torch import nn
import torch.functional as F

class Block(nn.Module):
    def __init__(
        self,
        d_in: int,
        d: int,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.BatchNorm1d(d_in),
            nn.Linear(d_in, d),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d, d_in)
        )
    def forward(self, x):
        return self.block(x)
