import pandas as pd
import numpy as np
import os
import json  # 新增：用于保存json文件
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

# ================= 配置部分 =================
DATASET_NAME = "student-mat"
FILE_PATH = 'data/student+performance/student-mat.csv'
OUTPUT_DIR = f'processed_data/{DATASET_NAME}'
TEST_SIZE = 0.2
RANDOM_SEED = 42

# ================= 1. 准备环境 =================
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"创建输出目录: {OUTPUT_DIR}")

# ================= 2. 加载数据 =================
print(f"正在读取文件: {FILE_PATH} ...")
# 确保文件路径存在，否则抛出更友好的错误
if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(f"找不到文件: {FILE_PATH}，请检查路径是否正确。")

df = pd.read_csv(FILE_PATH, sep=';')

# ================= 3. 分离标签 (Target) =================
target_col = 'G3'

if target_col not in df.columns:
    raise ValueError(f"数据集中找不到目标列 {target_col}")

y = df[target_col].values.astype(np.float32) # 统一转为float32
df_features = df.drop([target_col], axis=1)

# 删除强相关特征 G1, G2 (防止数据泄露)
drop_cols = ['G1', 'G2']
print(f"正在移除强相关特征: {drop_cols}")
df_features = df_features.drop(drop_cols, axis=1, errors='ignore')

# ================= 4. 区分数值与类别特征 =================
num_cols = df_features.select_dtypes(include=['int64', 'float64']).columns.tolist() # 转为list以便json序列化
cat_cols = df_features.select_dtypes(include=['object']).columns.tolist()

print(f"\n检测到 {len(num_cols)} 个数值特征: {num_cols}")
print(f"检测到 {len(cat_cols)} 个类别特征: {cat_cols}")

# ================= 5. 数据转换与编码 =================

# --- A. 处理数值数据 ---
X_num = df_features[num_cols].values.astype(np.float32)

# --- B. 处理类别数据 ---
encoder = OrdinalEncoder()
X_cat = encoder.fit_transform(df_features[cat_cols])
X_cat = X_cat.astype(np.float32)

# ================= 6. 切分训练集与验证集 =================
print(f"\n正在切分数据 (训练集: {1-TEST_SIZE:.0%}, 验证集: {TEST_SIZE:.0%}) ...")

X_num_train, X_num_val, \
X_cat_train, X_cat_val, \
y_train, y_val = train_test_split(
    X_num, X_cat, y, 
    test_size=TEST_SIZE, 
    random_state=RANDOM_SEED
)

# ================= 7. 保存为 .npy 文件 =================
print("\n正在保存文件到 .npy ...")

def save_npy(filename, data):
    path = os.path.join(OUTPUT_DIR, filename)
    np.save(path, data)
    print(f"已保存: {path} \t形状: {data.shape}")

save_npy('X_num_train.npy', X_num_train)
save_npy('X_num_val.npy', X_num_val)
save_npy('X_cat_train.npy', X_cat_train)
save_npy('X_cat_val.npy', X_cat_val)
save_npy('y_train.npy', y_train)
save_npy('y_val.npy', y_val)

# ================= 8. 生成并保存 info.json (新增部分) =================
print("\n正在生成 dataset info ...")

# 提取类别映射关系 (为了让 info.json 可读，需要将 numpy 数组转为 list)
cat_mapping = {}
for i, col in enumerate(cat_cols):
    # encoder.categories_[i] 是一个数组，包含该列所有唯一的字符串类别
    cat_mapping[col] = encoder.categories_[i].tolist()

info = {
    "dataset_name": DATASET_NAME,
    "task_type": "regression",  # G3 是连续数值，属于回归任务
    "n_samples": {
        "train": int(X_num_train.shape[0]),
        "val": int(X_num_val.shape[0]),
        "total": int(df.shape[0])
    },
    "n_features": {
        "numerical": int(X_num_train.shape[1]),
        "categorical": int(X_cat_train.shape[1]),
        "total": int(X_num_train.shape[1] + X_cat_train.shape[1])
    },
    "feature_names": {
        "numerical": num_cols,
        "categorical": cat_cols
    },
    "target_name": target_col,
    "files": {
        "X_num_train": "X_num_train.npy",
        "X_cat_train": "X_cat_train.npy",
        "y_train": "y_train.npy",
        "X_num_val": "X_num_val.npy",
        "X_cat_val": "X_cat_val.npy",
        "y_val": "y_val.npy"
    },
    "category_encoding_map": cat_mapping  # 记录具体的编码对应关系
}

json_path = os.path.join(OUTPUT_DIR, 'info.json')
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(info, f, indent=4, ensure_ascii=False)

print(f"已保存元数据: {json_path}")
print("\n处理完成！")