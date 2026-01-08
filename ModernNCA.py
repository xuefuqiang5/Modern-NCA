import torch
from torch import nn
from .Encoder import FeatureEncoder
from .Block import Block
from torch.utils.data import default_collate
from einops import rearrange, reduce, repeat

def SNS(dataset, sampling_rate=0.3):
    N = len(dataset)
    M = int(N * sampling_rate)
    indices = torch.randperm(N)[:M]
    raw_samples = [dataset[i] for i in indices]
    batch = default_collate(raw_samples)
    x_cat_sub, x_num_sub, y_sub = batch
    return x_cat_sub, x_num_sub, y_sub, indices

def SNS_fast(dataset, sampling_rate=0.3):
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
        return x_cat_sub, x_num_sub, y_sub, indices
    
    else:
        return SNS(dataset, sampling_rate)

def get_dist(x, sub_set, metric='euclidean'):
   
    if metric == 'euclidean':
        # 1. 维度对齐与广播 (Broadcasting) 准备
        # x:       [b, d] -> [b, 1, d] (为 sub_set 腾出中间维度)
        # sub_set: [n, d] -> [1, n, d] (为 x 腾出头部维度)
        x_expanded = rearrange(x, 'b d -> b 1 d')
        sub_set_expanded = rearrange(sub_set, 'n d -> 1 n d')
        
        # 2. 计算差值 (PyTorch 会自动广播: [b, 1, d] - [1, n, d] -> [b, n, d])
        # 这一步生成了所有成对的差向量
        diff = x_expanded - sub_set_expanded
        
        # 3. 计算 L2 范数: sqrt(sum(diff^2))
        # 使用 reduce 在最后一个维度 (d) 上求和
        # 'b n d -> b n' 清晰地表明我们将特征维度规约掉了，只剩下 [Batch, Num_Samples]
        dist_sq = reduce(diff**2, 'b n d -> b n', 'sum')
        
        # 开根号，加个微小值 eps 防止梯度爆炸
        return (dist_sq + 1e-8).sqrt()

    elif metric == 'cosine':
        # 额外提供：使用 einops 实现高效余弦距离 (1 - Cosine Similarity)
        from einops import einsum
        
        # 1. 归一化 (L2 Normalize)
        x_norm = torch.nn.functional.normalize(x, p=2, dim=1)
        sub_norm = torch.nn.functional.normalize(sub_set, p=2, dim=1)
        
        # 2. 矩阵乘法计算相似度
        # 'b d, n d -> b n' 直观表示：Batch与Dim点乘，Sample与Dim点乘，结果保留 Batch和Sample
        similarity = einsum(x_norm, sub_norm, 'b d, n d -> b n')
        
        # 3. 转换为距离
        return 1.0 - similarity

    else:
        raise ValueError("Unsupported metric")

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
def forward(self, x_cat, x_num): 
        if self.training:
            sub_x_cat, sub_x_num, sub_y, indics = SNS(self.entire, self.sampling_rate)
        else:
            sub_x_cat, sub_x_num, sub_y = self.entire[0], self.entire[1], self.entire[2]
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
            
        metric = get_dist(x, sub_x)
        
        alpha = nn.functional.softmax(-(metric**2), dim=-1)
        
        sub_y = rearrange(sub_y, "w -> 1 w")
        
        return torch.sum(alpha * sub_y, dim=-1)