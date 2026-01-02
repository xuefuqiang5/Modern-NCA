import torch
from torch import nn
from .Encoder import FeatureEncoder
from .Block import Block
from torch.utils.data import default_collate
from einops import rearrange, reduce, repeat
def SNS(dataset, sampling_rate=0.3):
    """
    针对 PyTorch Dataset 的随机邻域采样 (SNS)
    
    Args:
        dataset: 实现了 __getitem__ 返回 (x_cat, x_num, y) 的 Dataset
        sampling_rate (float): 采样比例
        
    Returns:
        x_cat_sub (Tensor): 采样后的类别特征 [M, ...]
        x_num_sub (Tensor): 采样后的数值特征 [M, ...]
        y_sub (Tensor): 采样后的标签 [M, ...]
        indices (Tensor): 对应的索引 [M]
    """
    # 1. 获取数据集长度
    N = len(dataset)
    M = int(N * sampling_rate)
    
    # 2. 生成随机索引 (通常 Dataset 索引在 CPU 上处理)
    # 这里的 indices 是乱序的
    indices = torch.randperm(N)[:M]
    
    # 3. 提取数据
    # 情况 A: 如果你的 Dataset 比较简单，数据都在内存里，这种列表推导式最通用
    raw_samples = [dataset[i] for i in indices]
    
    # 4. 堆叠 (Collate)
    # raw_samples 是一个 list: [(x_cat1, x_num1, y1), (x_cat2, x_num2, y2), ...]
    # default_collate 会把它转换成: (Batch_x_cat, Batch_x_num, Batch_y)
    # 且会自动处理 Tensor 堆叠和设备放置（默认 CPU Tensor）
    batch = default_collate(raw_samples)
    
    # 解包
    x_cat_sub, x_num_sub, y_sub = batch
    
    # 如果希望数据直接在 GPU 上（假设 dataset 本身是在 CPU 的）
    # 你可以在这里加 .cuda() 或者在外部处理
    # x_cat_sub = x_cat_sub.to(device) ...
    
    return x_cat_sub, x_num_sub, y_sub, indices

# ==========================================
# 优化版：如果你使用的是 TensorDataset
# ==========================================
def SNS_fast(dataset, sampling_rate=0.3):
    """
    如果 dataset 是 torch.utils.data.TensorDataset 或者是全量数据预加载在内存中的自定义 Dataset，
    直接切片比循环快得多。
    """
    N = len(dataset)
    M = int(N * sampling_rate)
    
    # 假设 dataset 内部存储了 tensors，且能在 GPU 上直接生成索引
    # 尝试获取 device，如果没有则默认为 cpu
    try:
        device = dataset.tensors[0].device 
    except AttributeError:
        device = 'cpu'
        
    indices = torch.randperm(N, device=device)[:M]
    
    # TensorDataset 支持直接访问 .tensors 属性
    # 这样避免了 Python for 循环的开销
    if hasattr(dataset, 'tensors'):
        # dataset.tensors 通常是 tuple (all_x_cat, all_x_num, all_y)
        x_cat_sub = dataset.tensors[0][indices]
        x_num_sub = dataset.tensors[1][indices]
        y_sub     = dataset.tensors[2][indices]
        return x_cat_sub, x_num_sub, y_sub, indices
    
    else:
        # 如果不是 TensorDataset，回退到通用方法
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
        sub_x_num, sub_x_cat, sub_y, indics = SNS(self.entire, self.sampling_rate)
        x = self.encoder(x_cat, x_num)
        sub_x = self.encoder(sub_x_num, sub_x_cat)
        sub_x.to(x_cat.device)
        for l in self.layers: 
            x = l(x)
            sub_x = l(sub_x)
        metric = get_dist(x, sub_x)
        # alpha.size = [batch_size, sub_y.size]
        alpha = nn.functional.softmax(-metric, dim=-1)
        sub_y = rearrange(sub_y, "w -> 1 w")
        return torch.sum(alpha * sub_y, dim=-1)

