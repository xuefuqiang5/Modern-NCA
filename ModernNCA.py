import torch
from torch import nn
from Encoder import FeatureEncoder
from Block import Block
from torch.utils.data import default_collate
from einops import rearrange, reduce, repeat
import torch.nn.functional as F

class ModernNCA(nn.Module): 
    def __init__(self, n_layers, n_num, cat_cardinalities, d_cat_embedding , d_num_embedding,  sampling_rate=0.3, tempature=1.0): 
        super().__init__()
        self.encoder = FeatureEncoder(n_num, cat_cardinalities, d_num_embedding, d_cat_embedding)
        d_out = self.encoder.d_out
        self.blocks = nn.ModuleList([
            Block(d_out, n_layers[i]) for i in range(len(n_layers))
        ])
        self.sampling_rate = sampling_rate
        self.T = tempature

    def _sns_sampling(self): 
        if self.training == True:
            return self._SNS(self.entire)
        else: 
            return (self.entire[0], self.entire[1], self.entire[2])

    def _SNS(self, dataset, sampling_rate=0.3):
        N = len(dataset[0])
        LIMIT = 1024 * 1024
        M = min(int(N * sampling_rate), LIMIT)
        
        try:
            device = dataset.tensors[0].device 
        except AttributeError:
            device = 'cpu'
            
        indices = torch.randperm(N, device=device)[:M]
        x_cat_sub = dataset[0][indices]
        x_num_sub = dataset[1][indices]
        y_sub     = dataset[2][indices]
        return x_cat_sub, x_num_sub, y_sub
        



    def forward(self, entire, x_cat, x_num, y): 
        candidate_x_cat, candidate_x_num, candidate_y = self._SNS(entire, self.sampling_rate) 
        candidate_x_cat = candidate_x_cat.to(x_cat.device)
        candidate_x_num = candidate_x_num.to(x_cat.device)
        candidate_y = candidate_y.to(x_cat.device)

        
        candidate_x = self.encoder(candidate_x_cat, candidate_x_num)
        x = self.encoder(x_cat, x_num)
        for l in self.blocks:
            candidate_x = l(candidate_x)
            x = l(x)
        if self.training: 
            assert y is not None
            candidate_x = torch.cat([x, candidate_x])
            candidate_y = torch.cat([y, candidate_y])
        distances = torch.cdist(x, candidate_x)
        distances = distances / self.T        
        if self.training:
            distances = distances.fill_diagonal_(torch.inf) 
        distances = F.softmax(-distances, dim=-1)
        logits = torch.mm(distances, candidate_y)
        return logits
