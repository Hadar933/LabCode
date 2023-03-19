from torch.utils.data import Dataset, DataLoader


class OnRunDataset(Dataset):
    def __init__(self, data_path: str, feature_lags: int, target_lags: int):
        self.data_path = data_path
        self.feature_lags = feature_lags
        self.target_lags = target_lags
