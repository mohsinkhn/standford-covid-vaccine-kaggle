"""Datasets required by pytorch dataloaders for competition data."""

import numpy as np
from torch.utils.data import Dataset

from constants import Token2Int


class RNAData(Dataset):
    def __init__(self, df, targets=["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]):
        self.df = df
        self.targets = targets
        self.prepare_inputs()

    def prepare_inputs(self):
        self.sequence = self.df["sequence"].apply(lambda x: [Token2Int.get(token) for token in x]).values
        self.structure = self.df["structure"].apply(lambda x: [Token2Int.get(token) for token in x]).values
        self.predicted_loop = (
            self.df["predicted_loop_type"].apply(lambda x: [Token2Int.get(token) for token in x]).values
        )
        self.stacked_inputs = np.dstack(
            (np.vstack(self.sequence), np.vstack(self.structure), np.vstack(self.predicted_loop))
        ).astype(int)
        self.target_values = np.dstack((np.vstack(self.df[col].values) for col in self.targets)).astype(np.float32)
        print(self.stacked_inputs.shape)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.stacked_inputs[idx], self.target_values[idx]
