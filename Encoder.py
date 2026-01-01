import torch
from torch import nn
from einops import rearrange, einsum


class PeriodicEmbedding(nn.Module): 
    def __init__(self, d_in, n_freq, freq_init_scale=0.1):
        super().__init__()
        self.frequencies = nn.Parameter(
            torch.randn(d_in, n_freq) * freq_init_scale
        )
    
    def forward(self, x): 
        x = rearrange(x, "... f -> ... f 1")
        x = x * self.frequencies
        sin_x = torch.sin(x)
        cos_x = torch.cos(x)
        return torch.cat([sin_x, cos_x], dim=-1)

class PLREncoder(nn.Module): 
    def __init(self, d_in, n_freq, d_embedding, freq_init_scale=0.1): 
        super().__init__()
        self.periodic_embedding = PeriodicEmbedding(d_in, n_freq)
        self.linear = nn.Linear(n_freq * 2, d_embedding)
    
    def forward(self, x): 
        x = self.periodic_embedding(x)
        return self.linear(x)

class FeatureEncoder(nn.Module): 
    def __init__(self, n_num_features, cat_cardinalitics, d_embedding, plr_freq=48, is_embedding=True):
        super().__init__()
        # plr embedding: x -> [x, sin(wx), cos(wx)] -> Linear -> ReLU
        self.plr = PLREncoder(n_num_features, plr_freq, d_embedding)
        self.is_embedding = is_embedding
        offset = torch.Tensor([0] + cat_cardinalitics[:-1]).cumsum(0)
        self.register_buffer("offset", offset)
        if is_embedding:
            self.embedding = nn.Embedding(sum(cat_cardinalitics), d_embedding)
        self.total_dim = sum(cat_cardinalitics)
        self.d_out = n_num_features * plr_freq + (len(cat_cardinalitics) * d_embedding)

    def forward(self, x_cat, x_num): 
        if self.is_embedding: 
            x_offset = x_cat + self.offset         
            x_cat = self.embedding(x_offset)
        else: 
            indices = self.offset + x_cat
            x_cat = torch.zeros(x_cat.size(0), self.total_dim, device=x_cat.device, dtype=torch.float32) 
            x_cat.scatter_(1, indices, 1.0)
        x_num = self.plr(x_num)
        x_num = rearrange(x_num, "... n l -> ... (n l)")
        return torch.cat([x_cat, x_num], dim=-1) 
