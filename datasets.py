"""Datasets required by pytorch dataloaders for competition data."""

import numpy as np
from torch.utils.data import Dataset
from constants import Mappings, FilePaths


TGT2ERR_COL = {"reactivity": "reactivity_error", "deg_Mg_50C": "deg_error_Mg_50C", "deg_Mg_pH10": "deg_error_Mg_pH10"}


class RNAData(Dataset):
    def __init__(self, df, targets=None, add_errors=False, add_bpp=False, FP=None, sig_factor=1.0):
        self.df = df
        self.targets = targets
        self.add_errors = add_errors
        self.add_bpp = add_bpp
        self.FP = FP
        self.sig_factor = sig_factor
        self.prepare_inputs()

    def prepare_inputs(self):
        self.sequence = (
            self.df["sequence"].apply(lambda x: [Mappings.sequence_token2int.get(token) for token in x]).values
        )
        self.structure = (
            self.df["structure"].apply(lambda x: [Mappings.structure_token2int.get(token) for token in x]).values
        )
        self.predicted_loop = (
            self.df["predicted_loop_type"].apply(lambda x: [Mappings.pl_token2int.get(token) for token in x]).values
        )
        if self.targets is not None:
            self.target_values = (
                np.dstack((np.vstack(self.df[col].values) for col in self.targets)).astype(np.float32).clip(-4, 4)
            )

        if self.add_errors:
            self.errors_sigma = (
                np.dstack((np.vstack(self.df[TGT2ERR_COL[col]].values) for col in self.targets))
                .astype(np.float32)
                .clip(-2, 2)
            )

        if self.add_bpp:
            self.ids = self.df["id"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        inputs = {
            "sequence": np.array(self.sequence[idx]),
            "structure": np.array(self.structure[idx]),
            "predicted_loop_type": np.array(self.predicted_loop[idx]),
        }
        if self.targets is not None:
            targets = self.target_values[idx]
        else:
            targets = [0]

        if self.add_errors:
            err_sig = self.errors_sigma[idx]
            targets += (np.random.gamma(shape=1.25, scale=1.0, size=err_sig.shape) - 1.25) * err_sig * self.sig_factor

        if self.add_bpp:
            rna_id = self.ids[idx]
            inputs["bpps"] = np.load(f"{self.FP.bpps_path}/{rna_id}.npy").astype("float32")
        return inputs, targets
