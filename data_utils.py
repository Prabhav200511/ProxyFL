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


def prepare_vanet_partitions(train_path, total_vehicles):
    """Split clients first, fit one scaler on training rows only, then scale.

    The contiguous per-device slices and local 80/20 split match the original
    simulation, while held-out client rows no longer influence preprocessing.
    """
    if not isinstance(total_vehicles, int) or total_vehicles <= 0:
        raise ValueError("total_vehicles must be a positive integer")
    frame = pd.read_csv(train_path)
    subset_size = len(frame) // total_vehicles
    raw_partitions = []
    for device_id in range(total_vehicles):
        start = device_id * subset_size
        end = (
            len(frame) if device_id == total_vehicles - 1
            else (device_id + 1) * subset_size
        )
        device_frame = frame.iloc[start:end].copy()
        if len(device_frame) < 2:
            raise ValueError(
                "each VANET partition needs at least two rows")
        n_test = max(1, int(len(device_frame) * 0.2))
        raw_partitions.append((
            device_frame.iloc[:-n_test].copy(),
            device_frame.iloc[-n_test:].copy(),
        ))

    training_union = pd.concat(
        [train for train, _ in raw_partitions], axis=0)
    scaler = StandardScaler().fit(training_union[VANETDataset.FEATURE_COLS])

    scaled_partitions = []
    for train, test in raw_partitions:
        train[VANETDataset.FEATURE_COLS] = scaler.transform(
            train[VANETDataset.FEATURE_COLS])
        test[VANETDataset.FEATURE_COLS] = scaler.transform(
            test[VANETDataset.FEATURE_COLS])
        scaled_partitions.append((train, test))
    return scaler, scaled_partitions


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
