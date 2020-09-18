"""Datasets required by pytorch dataloaders for competition data."""

import numpy as np
from torch.utils.data import Dataset
from constants import Mappings


class RNAData(Dataset):
    def __init__(self, df, targets=None):
        self.df = df
        self.targets = targets
        self.prepare_inputs()

    def prepare_inputs(self):
        self.sequence = (
            self.df["sequence"].apply(lambda x: [Mappings.sequence_token2int.get(token) for token in x]).values
        )
        self.structure = (
            self.df["structure"].apply(lambda x: [Mappings.structure_token2int.get(token) for token in x]).values
        )
        self.predicted_loop = (
            self.df["predicted_loop_type"]
            .apply(lambda x: [Mappings.pl_token2int.get(token) for token in x])
            .values
        )
        if self.targets is not None:
            self.target_values = np.dstack((np.vstack(self.df[col].values) for col in self.targets)).astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        inputs = {
            "sequence": np.array(self.sequence[idx]).reshape(-1, 1),
            "structure": np.array(self.structure[idx]).reshape(-1, 1),
            "predicted_loop_type": np.array(self.predicted_loop[idx]).reshape(-1, 1),
        }
        if self.targets is not None:
            return inputs, self.target_values[idx]
        return inputs, 0
