from torch.utils.data import Dataset
from typing import List
import os
import pandas as pd
import numpy as np
from utils import StandardScaler


class STFDataset(Dataset):
    def __init__(self, dataset_dir: str, train_val_test_ratio: List[float], input_len: int, output_len: int, mode: str = "train"):
        """
            Args:
                dataset_dir(str): Path to the dataset directory.
                train_val_test_ratio(List[float]): Ratio of train, validation and test datasets.
                input_len(int): Length of the input sequence.
                output_len(int): Length of the output sequence.
                steps_per_day(int): Number of steps per day.
        """
        self.dataset_dir = dataset_dir
        self.train_val_test_ratio = train_val_test_ratio
        self.input_len = input_len
        self.output_len = output_len
        self.mode = mode
        assert mode in ["train", "val", "test"], "mode should be one of ['train', 'val', 'test']"
        self.read_data()


    def read_data(self):
        # Read the data from the dataset directory and split it into train, validation and test datasets
        # based on the train_val_test_ratio.
        # The data should be stored in self.data as a list of tuples (input_sequence, output_sequence).
        # 读取数据集
        data_path = os.path.join(self.dataset_dir, "data.csv")
        df = pd.read_csv(data_path, index_col=0)  # (total_timesteps, num_nodes)
        df.index = pd.to_datetime(df.index, errors='coerce', format="mixed")  # 将索引转换为datetime格式
        train_ratio, val_ratio, test_ratio = self.train_val_test_ratio
        length, num_nodes = df.shape
        # 将数据集划分为训练、验证、测试集
        train_df = df.iloc[ : int(length * train_ratio)]
        val_df = df.iloc[int(length * train_ratio) - self.input_len : int(length * (train_ratio + val_ratio))]
        test_df = df.iloc[int(length * (train_ratio + val_ratio)) - self.input_len :]

        # 生成日历特征
        tod, dow = self.generate_calendar_features(df)        # 各 (T,)
        assert tod.shape == dow.shape == (length,), "日历特征长度与时间维不一致"
        cal = np.stack([tod, dow], axis=-1)                    # (T, 2)
        calendar = np.broadcast_to(cal[:, None, :], (length, num_nodes, 2))   # (T, N, 2)
        # 划分为训练、验证、测试
        train_calendar = calendar[ : int(length * train_ratio)]
        val_calendar = calendar[int(length * train_ratio) - self.input_len : int(length * (train_ratio + val_ratio))]
        test_calendar = calendar[int(length * (train_ratio + val_ratio)) - self.input_len :]

        
        # 数据标准化
        self.scaler = StandardScaler()
        train_data = self.scaler.fit_transform(train_df.values)
        val_data = self.scaler.transform(val_df.values)
        test_data = self.scaler.transform(test_df.values)

        # 根据模式选择数据集
        if self.mode == "train":
            selected_data = train_data.astype("float32", copy=False)
            selected_calendar = train_calendar
        elif self.mode == "val":
            selected_data = val_data.astype("float32", copy=False)
            selected_calendar = val_calendar
        else:
            selected_data = test_data.astype("float32", copy=False)
            selected_calendar = test_calendar
        
        # 保存选择的数据集数据
        selected_data = selected_data[:, :, None]  # (num_samples, num_nodes, 1)
        self.data = np.concatenate([selected_data, selected_calendar], axis=-1).astype(np.float32, copy=False)    # (num_samples, num_nodes, 3)


    def generate_calendar_features(self, df):
        # 生成日历特征
        # tod is a fraction of one day, e.g. 5-minute data increases by 1/288;
        # dow follows pandas' Monday=0,...,Sunday=6 convention.
        seconds = (
            df.index.hour.to_numpy() * 3600
            + df.index.minute.to_numpy() * 60
            + df.index.second.to_numpy()
            + df.index.microsecond.to_numpy() / 1_000_000.0
        )
        tod = (seconds / 86_400.0).astype("float32")
        dow = df.index.dayofweek.to_numpy(dtype="int")
        return tod, dow
    
    
    def __len__(self):
        return max(0, len(self.data) - self.input_len - self.output_len + 1)


    def __getitem__(self, idx):
        x = self.data[idx : idx + self.input_len]  # (input_len, num_nodes, 3)
        y = self.data[idx + self.input_len : idx + self.input_len + self.output_len]  # (output_len, num_nodes, 3)
        return x, y