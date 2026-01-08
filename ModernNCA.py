import torch
from torch import nn
from .Encoder import FeatureEncoder
from .Block import Block
from torch.utils.data import default_collate
from einops import rearrange, reduce, repeat

def SNS(dataset, sampling_rate=0.3):
    N = len(dataset)
    M = int(N * sampling_rate)
    
    try:
        device = dataset.tensors[0].device 
    except AttributeError:
        device = 'cpu'
        
    indices = torch.randperm(N, device=device)[:M]
    
    if hasattr(dataset, 'tensors'):
        x_cat_sub = dataset.tensors[0][indices]
        x_num_sub = dataset.tensors[1][indices]
        y_sub     = dataset.tensors[2][indices]
        return x_cat_sub, x_num_sub, y_sub
    
    else:
        return SNS(dataset, sampling_rate)

class ModernNCA(nn.Module): 
    def __init__(self, entire, n_layers, n_num, cat_cardinalitics, d_cat_embedding , d_num_embedding, n_freq=48, scale=0.1, sampling_rate=0.3): 
        super().__init__()
        self.encoder = FeatureEncoder(n_num, cat_cardinalitics, d_num_embedding, d_cat_embedding, n_freq, is_embedding=False)
        n_layers = [self.encoder.d_out] + n_layers
        self.layers = nn.ModuleList([
            Block(n_layers[i-1], n_layers[i]) for i in range(1, len(n_layers))
        ])
        self.entire = entire
        self.sampling_rate = sampling_rate

    def _sns_sampling(self): 
        if self.training == True:
            return SNS(self.entire)
        else: 
            return (self.entire[0], self.entire[1], self.entire[2])

    def _get_dist(x, sub_set, metric='euclidean'):
    
        if metric == 'euclidean':
            x_expanded = rearrange(x, 'b d -> b 1 d')
            sub_set_expanded = rearrange(sub_set, 'n d -> 1 n d')
            
            diff = x_expanded - sub_set_expanded
            
            dist_sq = reduce(diff**2, 'b n d -> b n', 'sum')
            return (dist_sq + 1e-8).sqrt()
        elif metric == 'cosine':
            from einops import einsum
            x_norm = torch.nn.functional.normalize(x, p=2, dim=1)
            sub_norm = torch.nn.functional.normalize(sub_set, p=2, dim=1)
            similarity = einsum(x_norm, sub_norm, 'b d, n d -> b n')
            return 1.0 - similarity

        else:
            raise ValueError("Unsupported metric")


    def forward(self, x_cat, x_num): 
        sub_x_cat, sub_x_num, sub_y= self._sns_sampling()
        device = x_cat.device
        sub_x_num = sub_x_num.to(device)
        sub_x_cat = sub_x_cat.to(device)
        sub_y = sub_y.to(device)

        # 3. 编码 Query (当前输入) 和 Support (参考集)
        x = self.encoder(x_cat, x_num)
        sub_x = self.encoder(sub_x_cat, sub_x_num)
        
        for l in self.layers: 
            x = l(x)
            sub_x = l(sub_x)
            
        metric = self._get_dist(x, sub_x)
        
        alpha = nn.functional.softmax(-(metric**2), dim=-1)
        
        sub_y = rearrange(sub_y, "w -> 1 w")
        
        return torch.sum(alpha * sub_y, dim=-1)