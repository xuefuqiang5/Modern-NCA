import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.optim as optim
from ModernNCA import ModernNCA
import numpy as np
import json
import os
import yaml
import argparse
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

class TabularDataset(Dataset):
    def __init__(self, x_cat, x_num, y):
        self.x_num = x_num
        self.x_cat = x_cat
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x_cat[idx], self.x_num[idx], self.y[idx]

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
        
        scaler = StandardScaler()
        x_num = scaler.fit_transform(x_num_np)
        x_num = torch.from_numpy(x_num).float()
        
        x_cat = torch.from_numpy(x_cat_np).long()
        
        if info['task_type'] == 'regression':
            y = torch.from_numpy(y_np).float()
            if len(y.shape) == 1:
                y = y.unsqueeze(1) # (N) -> (N, 1)
        else:
            y = torch.from_numpy(y_np).long()
            
        return TabularDataset(x_cat, x_num, y)

    train_dataset = load_part('train')
    val_dataset = load_part('val')
    
    print(f"加载完成 -> 训练集: {len(train_dataset)} 样本, 验证集: {len(val_dataset)} 样本")
    
    return train_dataset, val_dataset

def get_dataloader(dataset, batch_size, is_shuffle=True): 
    return DataLoader(dataset, batch_size, is_shuffle)

def get_args():
    parser = argparse.ArgumentParser(description="ModernNCA Hyperparameters Configuration")
    parser.add_argument('--config', type=str, default='./config.yaml', help='Path to YAML config file')
    parser.add_argument('--n_layers', type=int, nargs='+', default=[256, 128],
                        help='--n_layers 256 128')
    
    parser.add_argument('--d_cat_embedding', type=int, default=32,
                        help='Cat Embedding Dimension')
    
    parser.add_argument('--d_num_embedding', type=int, default=32,
                        help='Num Embedding Dimension')
    
    parser.add_argument('--n_freq', type=int, default=48,
                        help='Feature Encoder Frequency')

    parser.add_argument('--sampling_rate', type=float, default=0.3,
                        help='Sampling rate for Stochastic Neighborhood Sampling')
    
    parser.add_argument('--scale', type=float, default=0.1,
                        help='Scale factor')
    
    parser.add_argument('--metric', type=str, default='euclidean', choices=['euclidean', 'cosine'],
                        help='Distance metric')

    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch Size')
    
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of Epochs')
    
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning Rate')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='Random Seed')
    
    parser.add_argument('--data_path', type=str, default='./data/dataset.csv',
                        help='datafile path')

    args = parser.parse_args()
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    flat_config = {}
    for v in config.values():
        flat_config.update(v)

    for key, value in flat_config.items():
        if not hasattr(args, key) or getattr(args, key) is None:
            setattr(args, key, value)
    return args

def get_model(args, info_path):
        
    with open(info_path, 'r', encoding='utf-8') as f:
        info = json.load(f)

        n_num = info['n_features']['numerical']
        cat_cardinalities = [len(categories) for categories in info['category_encoding_map'].values()]
    print(f"the paramater of model: n_num={n_num}, cat_cardinalities={cat_cardinalities}")
    model = ModernNCA(
        n_layers=args.n_layers,
        n_num=n_num,
        cat_cardinalities=cat_cardinalities,
        d_cat_embedding=args.d_cat_embedding,
        d_num_embedding=args.d_num_embedding,
        sampling_rate=args.sampling_rate,
        tempature=args.tempature
    )
    
    return model   

def train_one_epoch(model, optimizer, data_loader, entire, criterion, device):
    model.train()  # 切换到训练模式 (开启 Dropout, BatchNorm 更新等)
    total_loss = 0.0
    
    pbar = tqdm(data_loader, desc="Training", leave=False)
    
    for x_cat, x_num, y in pbar:
        x_num = x_num.to(device)
        x_cat = x_cat.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        predictions = model(entire, x_cat, x_num, y)
        loss = criterion(predictions, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
    avg_loss = total_loss / len(data_loader)
    return avg_loss

def evaluate(model, data_loader, entire, criterion, device, task_type):
    model.eval()  # 切换到评估模式
    total_loss = 0.0
    all_preds = []
    all_targets = []
    with torch.no_grad():  # 不计算梯度，节省显存
        for x_cat, x_num, y in data_loader:
            x_num = x_num.to(device)
            x_cat = x_cat.to(device)
            y = y.to(device)
            predictions = model(entire, x_cat, x_num, y)
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
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

if __name__ == "__main__": 
    train_dataset, val_dataset = get_dataset("processed_data/Student-Test-Scores")
    train_dataloader = get_dataloader(train_dataset, 32)
    val_dataloader = get_dataloader(val_dataset, 32)
    args = get_args()
    model = get_model(args, "processed_data/Student-Test-Scores/info.json")
    optimizer = optim.Adam(
        model.parameters(), 
        lr = args.lr
    )
    entire_data = (
        train_dataset.x_cat, 
        train_dataset.x_num,
        train_dataset.y
    )

    device = get_device()
    print(f"device: {device}")
    model = model.to(device)
    for epoch in range(args.epochs): 
        train_one_epoch(model, optimizer, train_dataloader,entire_data, nn.MSELoss(), device)
        with open("train.log", "w+", encoding="utf-8") as f: 
            f.write(evaluate(model, val_dataloader, entire_data, nn.MSELoss, device, "regression"))

    