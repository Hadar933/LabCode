from typing import Dict
from preprocessor import Preprocess
from encoders import Encoder
import pandas as pd
from torch.utils.data import Dataset
import torch


def tensor_mb_size(v: torch.Tensor):
    return v.nelement() * v.element_size() / 1_000_000


class WingDataset(Dataset):
    def __init__(self, data_path: str, feature_lags: int, target_lags: int):
        self.data_path = data_path
        self.feature_lags = feature_lags
        self.target_lags = target_lags
        self.raw_data: pd.DataFrame = self.read_data()
        self.preprocessed_data: torch.Tensor = None
        self.encoded_data: pd.DataFrame = None
        self.feature_cols: Dict[str, int] = {}
        self.target_cols: Dict[str, int] = {}

    def read_data(self) -> pd.DataFrame:
        if self.data_path.endswith(".csv"):
            try:
                result = pd.read_csv(self.data_path, index_col=0, parse_dates=['ts'])
            except ValueError:
                result = pd.read_csv(self.data_path, index_col=0, compression="gzip")

        elif self.data_path.endswith(".gz"):
            try:
                result = pd.read_csv(self.data_path, index_col=0, parse_dates=['ts'], compression="gzip")
            except ValueError:
                result = pd.read_csv(self.data_path, index_col=0, compression="gzip")

        elif self.data_path.endswith(".pkl"):
            result = pd.read_pickle(self.data_path)
        else:
            raise ValueError(f"[Reader] File {self.data_path} has wrong format")
        return result

    def encode_data(self):
        return self.preprocessed_data

    def __len__(self):
        return len(self.preprocessed_data) - self.feature_lags

    def __getitem__(self, idx):
        features_window = self.preprocessed_data[idx: idx + self.feature_lags, list(self.feature_cols.values())]
        target_window = self.preprocessed_data[idx: idx + self.target_lags, list(self.target_cols.values())]
