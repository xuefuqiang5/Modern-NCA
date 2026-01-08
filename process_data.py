import pandas as pd
import numpy as np
import os
import json
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

class TabularDataProcessor:
    def __init__(self, 
                 file_path, 
                 target_col, 
                 task_type='regression', 
                 drop_cols=None, 
                 sep=',', 
                 dataset_name=None,
                 test_size=0.2, 
                 random_seed=42,
                 output_root='processed_data'):
        """
        通用的表格数据预处理类
        
        :param file_path: CSV文件路径
        :param target_col: 预测的目标列名
        :param task_type: 'regression' (回归) 或 'classification' (分类)
        :param drop_cols: 需要丢弃的列名列表 (list)
        :param sep: CSV分隔符
        :param test_size: 验证集比例
        :param random_seed: 随机种子
        :param output_root: 输出根目录
        """
        self.file_path = file_path
        self.target_col = target_col
        self.task_type = task_type
        self.drop_cols = drop_cols if drop_cols else []
        self.sep = sep
        self.test_size = test_size
        self.random_seed = random_seed
        
        # 自动推导数据集名称 (文件名去掉扩展名)
        if dataset_name is None:
            self.dataset_name = os.path.splitext(os.path.basename(file_path))[0]
            self.output_dir = os.path.join(output_root, self.dataset_name)
        else: 
            self.dataset_name = dataset_name
            self.output_dir = os.path.join(output_root, self.dataset_name)
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def process(self):
        print(f"[{self.dataset_name}] 开始处理...")
        
        # 1. 加载数据
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"文件未找到: {self.file_path}")
        
        # 尝试读取，处理常见错误
        try:
            df = pd.read_csv(self.file_path, sep=self.sep)
        except Exception as e:
            raise ValueError(f"读取CSV失败，请检查分隔符是否正确。错误信息: {e}")

        # 检查目标列
        if self.target_col not in df.columns:
            raise ValueError(f"目标列 '{self.target_col}' 不在数据集中。可用列: {list(df.columns)}")

        # 2. 丢弃不需要的列
        if self.drop_cols:
            print(f"正在移除指定列: {self.drop_cols}")
            df = df.drop(self.drop_cols, axis=1, errors='ignore')

        # 3. 简单的缺失值填充 (为了泛化性，简单策略)
        # 数值填均值，类别填 "Missing"
        print("检查并处理缺失值...")
        for col in df.columns:
            if df[col].isnull().any():
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].mean())
                else:
                    df[col] = df[col].fillna("Missing")

        # 4. 分离特征和标签
        y_raw = df[self.target_col]
        df_features = df.drop([self.target_col], axis=1)

        # 5. 处理标签 (Target)
        y, target_mapping = self._process_target(y_raw)

        # 6. 处理特征 (Features)
        X_num, X_cat, num_cols, cat_cols, feature_encoder = self._process_features(df_features)

        # 7. 切分数据集
        print(f"切分数据集 (验证集: {self.test_size})...")
        X_num_train, X_num_val, X_cat_train, X_cat_val, y_train, y_val = train_test_split(
            X_num, X_cat, y, 
            test_size=self.test_size, 
            random_state=self.random_seed,
            stratify=y if self.task_type == 'classification' else None # 分类任务保持类别分布
        )

        # 8. 保存 .npy 文件
        self._save_npy_files(X_num_train, X_num_val, X_cat_train, X_cat_val, y_train, y_val)

        # 9. 生成并保存 info.json
        self._save_info_json(df, num_cols, cat_cols, X_num_train, X_cat_train, feature_encoder, target_mapping)
        
        print(f"[{self.dataset_name}] 处理完成！数据保存在: {self.output_dir}")

    def _process_target(self, y_raw):
        """根据任务类型处理标签"""
        target_mapping = None
        
        if self.task_type == 'regression':
            # 回归任务：转为 float32
            y = y_raw.values.astype(np.float32)
        
        elif self.task_type == 'classification':
            # 分类任务：使用 LabelEncoder 转为 0, 1, 2...
            le = LabelEncoder()
            y = le.fit_transform(y_raw)
            y = y.astype(np.int64) # 分类标签通常用 long/int64
            # 记录类别映射，方便反查
            target_mapping = {str(k): int(v) for k, v in zip(le.classes_, range(len(le.classes_)))}
            print(f"分类任务检测到 {len(le.classes_)} 个类别。")
        else:
            raise ValueError("task_type 必须是 'regression' 或 'classification'")
            
        return y, target_mapping

    def _process_features(self, df_features):
        """区分数值和类别特征并编码"""
        # 自动识别类型
        num_cols = df_features.select_dtypes(include=['int64', 'float64']).columns.tolist()
        cat_cols = df_features.select_dtypes(include=['object', 'bool', 'category']).columns.tolist()

        print(f"数值特征: {len(num_cols)} 个, 类别特征: {len(cat_cols)} 个")

        # --- A. 数值特征 ---
        if num_cols:
            X_num = df_features[num_cols].values.astype(np.float32)
        else:
            # 如果没有数值特征，创建一个空的 placeholder，防止报错
            X_num = np.zeros((len(df_features), 0), dtype=np.float32)

        # --- B. 类别特征 ---
        feature_encoder = None
        if cat_cols:
            # 强制将所有类别转为字符串，防止混合类型报错
            df_cat = df_features[cat_cols].astype(str)
            
            feature_encoder = OrdinalEncoder()
            X_cat = feature_encoder.fit_transform(df_cat)
            X_cat = X_cat.astype(np.float32)
        else:
            # 如果没有类别特征
            X_cat = np.zeros((len(df_features), 0), dtype=np.float32)
            
        return X_num, X_cat, num_cols, cat_cols, feature_encoder

    def _save_npy_files(self, X_num_train, X_num_val, X_cat_train, X_cat_val, y_train, y_val):
        files = {
            'X_num_train.npy': X_num_train,
            'X_num_val.npy': X_num_val,
            'X_cat_train.npy': X_cat_train,
            'X_cat_val.npy': X_cat_val,
            'y_train.npy': y_train,
            'y_val.npy': y_val
        }
        for name, data in files.items():
            np.save(os.path.join(self.output_dir, name), data)

    def _save_info_json(self, df, num_cols, cat_cols, X_num_train, X_cat_train, feature_encoder, target_mapping):
        # 提取类别特征的映射字典
        cat_encoding_map = {}
        if feature_encoder:
            for i, col in enumerate(cat_cols):
                # 将 numpy array 转为 list 以便 JSON 序列化
                cat_encoding_map[col] = feature_encoder.categories_[i].tolist()

        info = {
            "dataset_name": self.dataset_name,
            "task_type": self.task_type,
            "n_samples": {
                "train": int(X_num_train.shape[0]),
                "val": int(df.shape[0]-X_num_train.shape[0]), # 使用对应的部分大小
                "total": int(df.shape[0])
            },
            "n_features": {
                "numerical": len(num_cols),
                "categorical": len(cat_cols),
                "total": len(df.columns) - 1 # 减去target
            },
            "feature_names": {
                "numerical": num_cols,
                "categorical": cat_cols
            },
            "target_name": self.target_col,
            "target_mapping": target_mapping, # 如果是分类任务，这里会有 label map
            "files": {
                "X_num_train": "X_num_train.npy",
                "X_cat_train": "X_cat_train.npy",
                "y_train": "y_train.npy",
                "X_num_val": "X_num_val.npy",
                "X_cat_val": "X_cat_val.npy",
                "y_val": "y_val.npy"
            },
            "category_encoding_map": cat_encoding_map
        }
        
        with open(os.path.join(self.output_dir, 'info.json'), 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=4, ensure_ascii=False)

# ================= 命令行调用接口 =================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="通用表格数据预处理工具")
    
    # 必须参数
    parser.add_argument("--file", type=str, required=True, help="CSV文件路径")
    parser.add_argument("--target", type=str, required=True, help="目标列名")
    
    # 可选参数
    parser.add_argument("--type", type=str, default="regression", choices=["regression", "classification"], help="任务类型")
    parser.add_argument("--sep", type=str, default=",", help="CSV分隔符 (例如: ',' 或 ';')")
    parser.add_argument("--drop", type=str, nargs='+', help="需要丢弃的列名列表 (空格分隔)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    args_list = [
        "--file", "data/playground-series-s6e1/train.csv", 
        "--target", "exam_score",
        "--type", "regression", 
        "--sep", ",", 
        "--seed", "42",
    ] 
    args = parser.parse_args(args_list)
    # 实例化并运行
    processor = TabularDataProcessor(
        file_path=args.file,
        target_col=args.target,
        task_type=args.type,
        drop_cols=args.drop,
        sep=args.sep,
        random_seed=args.seed,
        dataset_name="Student-Test-Scores"
    )
    
    processor.process()