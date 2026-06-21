import os
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class Dataset_Solar_Calendar(Dataset):
    """Solar benchmark with a position-derived intraday phase index."""

    def __init__(
        self,
        root_path,
        flag="train",
        size=None,
        features="M",
        data_path="solar_AL.txt",
        target="OT",
        scale=True,
        timeenc=0,
        freq="t",
        period=144,
    ):
        self.seq_len, self.label_len, self.pred_len = size
        self.set_type = {"train": 0, "val": 1, "test": 2}[flag]
        self.scale = scale
        self.period = period
        self.root_path = root_path
        self.data_path = data_path
        self._read_data()

    def _read_data(self):
        rows = []
        with open(
            os.path.join(self.root_path, self.data_path),
            "r",
            encoding="utf-8",
        ) as handle:
            for line in handle:
                rows.append(np.asarray(line.rstrip().split(","), dtype=np.float32))
        raw = np.stack(rows)

        num_train = int(len(raw) * 0.7)
        num_test = int(len(raw) * 0.2)
        num_valid = len(raw) - num_train - num_test
        border1s = [
            0,
            num_train - self.seq_len,
            len(raw) - num_test - self.seq_len,
        ]
        border2s = [
            num_train,
            num_train + num_valid,
            len(raw),
        ]
        self.border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        self.scaler = StandardScaler()
        if self.scale:
            self.scaler.fit(raw[:num_train])
            raw = self.scaler.transform(raw)

        self.data_x = raw[self.border1:border2]
        self.data_y = raw[self.border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((self.seq_len, 1))
        seq_y_mark = torch.zeros((self.label_len + self.pred_len, 1))

        # s_end is the first forecast timestamp in the local split.
        phase = torch.tensor(
            (self.border1 + s_end) % self.period, dtype=torch.long
        )
        return seq_x, seq_y, seq_x_mark, seq_y_mark, phase

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
