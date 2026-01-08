    
import torch
from torch.utils.data import Dataset, DataLoader
from .ModernNCA import ModernNCA
import numpy as np
import json
import os
from tqdm import tqdm

class TabularDataset(Dataset):
    def __init__(self, x_num, x_cat, y):
        self.x_num = x_num
        self.x_cat = x_cat
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x_num[idx], self.x_cat[idx], self.y[idx]

def get_dataset(data_dir):
    json_path = os.path.join(data_dir, 'info.json')
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"元数据文件不存在: {json_path}")
        
    with open(json_path, 'r', encoding='utf-8') as f:
        info = json.load(f)
    
    print(f"正在加载数据集: {info['dataset_name']} (任务类型: {info['task_type']})")
    
    files = info['files']
    
    def load_part(prefix):
        path_num = os.path.join(data_dir, files[f'X_num_{prefix}'])
        path_cat = os.path.join(data_dir, files[f'X_cat_{prefix}'])
        path_y = os.path.join(data_dir, files[f'y_{prefix}'])
        
        x_num_np = np.load(path_num)
        x_cat_np = np.load(path_cat)
        y_np = np.load(path_y)
        
        x_num = torch.from_numpy(x_num_np).float()
        
        x_cat = torch.from_numpy(x_cat_np).long()
        
        if info['task_type'] == 'regression':
            y = torch.from_numpy(y_np).float()
            if len(y.shape) == 1:
                y = y.unsqueeze(1) # (N) -> (N, 1)
        else:
            y = torch.from_numpy(y_np).long()
            
        return TabularDataset(x_num, x_cat, y)

    train_dataset = load_part('train')
    val_dataset = load_part('val')
    
    print(f"加载完成 -> 训练集: {len(train_dataset)} 样本, 验证集: {len(val_dataset)} 样本")
    
    return train_dataset, val_dataset, info

def get_dataloader(dataset, batch_size, is_shuffle=True): 
    return DataLoader(dataset, batch_size, is_shuffle)

def get_model(info, train_dataset, config=None):
    default_config = {
        'n_layers': 2,
        'd_cat_embedding': 8,   # 每个类别特征映射为 8 维
        'd_num_embedding': 16,  # 每个数值特征映射为 16 维
        'n_freq': 48,
        'scale': 0.1,
        'sampling_rate': 0.5    # 小数据集建议设大一点，甚至 1.0
    }
    
    if config:
        default_config.update(config)
    
    cfg = default_config
    n_num = info['n_features']['numerical']
    cat_cardinalities = [len(categories) for categories in info['category_encoding_map'].values()]
    print(f"检测到模型参数: n_num={n_num}, cat_cardinalities={cat_cardinalities}")
    entire_data = (
        train_dataset.x_num, 
        train_dataset.x_cat, 
        train_dataset.y
    )
    model = ModernNCA(
        entire=entire_data,
        n_layers=cfg['n_layers'],
        n_num=n_num,
        cat_cardinalities=cat_cardinalities,
        d_cat_embedding=cfg['d_cat_embedding'],
        d_num_embedding=cfg['d_num_embedding'],
        n_freq=cfg['n_freq'],
        scale=cfg['scale'],
        sampling_rate=cfg['sampling_rate']
    )
    
    return model   

def train_one_epoch(model, optimizer, data_loader, criterion, device):
    model.train()  # 切换到训练模式 (开启 Dropout, BatchNorm 更新等)
    total_loss = 0.0
    
    pbar = tqdm(data_loader, desc="Training", leave=False)
    
    for x_num, x_cat, y in pbar:
        x_num = x_num.to(device)
        x_cat = x_cat.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        predictions = model(x_num, x_cat)
        loss = criterion(predictions, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
    avg_loss = total_loss / len(data_loader)
    return avg_loss

def evaluate(model, data_loader, criterion, device, task_type):
    model.eval()  # 切换到评估模式
    total_loss = 0.0
    all_preds = []
    all_targets = []
    with torch.no_grad():  # 不计算梯度，节省显存
        for x_num, x_cat, y in data_loader:
            x_num = x_num.to(device)
            x_cat = x_cat.to(device)
            y = y.to(device)
            
            predictions = model(x_num, x_cat)
            loss = criterion(predictions, y)
            total_loss += loss.item()
            
            if task_type == 'classification':
                preds = torch.argmax(predictions, dim=1)
            else:
                preds = predictions
            
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            
    avg_loss = total_loss / len(data_loader)
    
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    metric_score = 0.0
    metric_name = ""
    
    if task_type == 'regression':
        mse = np.mean((all_preds - all_targets) ** 2)
        metric_score = np.sqrt(mse)
        metric_name = "RMSE"
    else:
        correct = (all_preds == all_targets).sum()
        metric_score = correct / len(all_targets)
        metric_name = "Accuracy"
        
    return avg_loss, metric_score, metric_name

