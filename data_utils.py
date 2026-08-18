# data_utils.py — Shared dataset utilities for VANET IDS
from functools import lru_cache

import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class VANETDataset(Dataset):
    """PyTorch Dataset for VANET IDS telemetry data (VeReMi features)."""

    FEATURE_COLS = ['velocity_x', 'velocity_y', 'constant_offset_check', 'total_displacement']
    TARGET_COL = 'attacktype'

    def __init__(self, dataframe):
        features = dataframe[self.FEATURE_COLS].values
        targets = dataframe[self.TARGET_COL].values

        self.x = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


@lru_cache(maxsize=1)
def get_vanet_scaler(train_path='Main_data_shuffled.csv'):
    """Fit the canonical feature scaler used by every VANET participant.

    Federated averaging assumes each client model receives the same feature
    representation. Fitting a scaler per client makes their weights
    incompatible, so this scaler is fit once on the common training data and
    reused for client and evaluation data.
    """
    train_df = pd.read_csv(train_path, usecols=VANETDataset.FEATURE_COLS)
    return StandardScaler().fit(train_df)
