from .data import STFDataset
from typing import List
from torch.utils.data import DataLoader

def get_dataloader(dataset_dir: str, train_val_test_ratio: List[float], input_len: int, output_len: int, batch_size: int, shuffle: bool=True, num_workers: int=0):
    """
    Returns a DataLoader for the given dataset.

    Args:
        dataset_dir (str): The directory containing the dataset.
        train_val_test_ratio (List[float]): The ratio of training, validation, and test samples.
        input_len (int): The length of the input sequence.
        output_len (int): The length of the output sequence.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the data at every epoch.
        num_workers (int): How many subprocesses to use for data loading.
    Returns:
        torch.utils.data.DataLoader: DataLoader for the given dataset.
    """
    data = {}
    # 创建dataset
    train_dataset = STFDataset(dataset_dir, train_val_test_ratio=train_val_test_ratio, input_len=input_len, output_len=output_len, mode="train")
    val_dataset = STFDataset(dataset_dir, train_val_test_ratio=train_val_test_ratio, input_len=input_len, output_len=output_len, mode="val")
    test_dataset = STFDataset(dataset_dir, train_val_test_ratio=train_val_test_ratio, input_len=input_len, output_len=output_len, mode="test")
    # 构建dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    # 获取Scaler, 便于后续恢复原始尺度
    scaler = train_dataset.scaler
    # 保存为字典
    data['train_loader'] = train_loader
    data['val_loader'] = val_loader
    data['test_loader'] = test_loader
    data['scaler'] = scaler
    return data