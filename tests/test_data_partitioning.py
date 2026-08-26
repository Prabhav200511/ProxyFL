import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from data_utils import VANETDataset, prepare_vanet_partitions


class DataPartitioningTests(unittest.TestCase):
    def test_scaler_uses_only_union_of_local_training_rows(self):
        columns = VANETDataset.FEATURE_COLS
        frame = pd.DataFrame({
            columns[0]: [0, 2, 1000, 4, 6, 2000],
            columns[1]: [10, 12, 1010, 14, 16, 2010],
            columns[2]: [20, 22, 1020, 24, 26, 2020],
            columns[3]: [30, 32, 1030, 34, 36, 2030],
            VANETDataset.TARGET_COL: [0, 1, 0, 1, 0, 1],
        })
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "train.csv")
            frame.to_csv(path, index=False)
            scaler, partitions = prepare_vanet_partitions(
                path, total_vehicles=2)
            scaler_again, partitions_again = prepare_vanet_partitions(
                path, total_vehicles=2)

        expected_train = frame.loc[[0, 1, 3, 4], columns]
        np.testing.assert_allclose(
            scaler.mean_, expected_train.mean(axis=0).to_numpy())
        self.assertFalse(np.allclose(
            scaler.mean_, frame[columns].mean(axis=0).to_numpy()))
        np.testing.assert_allclose(scaler.mean_, scaler_again.mean_)
        for (train, test), (train_again, test_again) in zip(
                partitions, partitions_again):
            pd.testing.assert_frame_equal(train, train_again)
            pd.testing.assert_frame_equal(test, test_again)


if __name__ == "__main__":
    unittest.main()
