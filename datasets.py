"""Datasets required by pytorch dataloaders for competition data."""

from collections import defaultdict
import json

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from constants import Mappings, FilePaths


TGT2ERR_COL = {"reactivity": "reactivity_error", "deg_Mg_50C": "deg_error_Mg_50C", "deg_Mg_pH10": "deg_error_Mg_pH10"}


def match_pair(structure):
    pair = [-1] * len(structure)
    pair_no = -1

    pair_no_stack = []
    for i, c in enumerate(structure):
        if c == "(":
            pair_no += 1
            pair[i] = pair_no
            pair_no_stack.append(pair_no)
        elif c == ")":
            pair[i] = pair_no_stack.pop()
    # twin_seq = [sequence[idx] if idx >= 0 else 0 for idx in pair]
    return pair


class RNAAugData(Dataset):
    def __init__(self, df, targets=None, target_aug=False, augment_strucures=False, aug_data_sources=None, bpps_path="data/bpps"):
        self.df = df
        self.targets = targets
        self.num_targets = len(targets) if targets is not None else 3
        self.bpps_path = bpps_path
        self.target_aug = target_aug
        self.augment_strucures = augment_strucures
        self.aug_data_sources = aug_data_sources
        self.aug_data = None
        if self.augment_strucures:
            self.aug_data = pd.concat([pd.read_csv(src) for src in aug_data_sources])
        self.prepare_inputs()  # Tokenize stuff and keep as numpy array for quick batch loading

    def aggregate_options(self, col, token2int):
        id2arr = defaultdict(list)
        for (id, arr) in self.df[["id", col]].values.tolist():
            id2arr[id].append([token2int.get(token, 0) for token in arr])
        if self.aug_data is not None:
            for (id, arr) in self.aug_data[["id", col]].values.tolist():
                id2arr[id].append([token2int.get(token, 0) for token in arr])
        return id2arr

    @staticmethod
    def base_pair_index_to_sequence(df):  # From base pair index, get actual index
        df["pair_sequence"] = df[["sequence", "pair_index"]].apply(
            lambda x: [x["sequence"][idx] if idx >= 0 else "0" for idx in x["pair_index"]], axis=1
        )
        return df

    def get_match_pairs(self):  # Get index of base pairs
        self.df["pair_index"] = self.df["structure"].apply(lambda x: match_pair(x))
        self.df = self.base_pair_index_to_sequence(self.df)
        if self.aug_data is not None:
            self.aug_data["pair_index"] = self.aug_data["structure"].apply(lambda x: match_pair(x))
            self.aug_data = self.base_pair_index_to_sequence(self.aug_data)

    def prepare_inputs(self):
        self.get_match_pairs()
        self.sequence = self.aggregate_options("sequence", Mappings.sequence_token2int)
        self.structure = self.aggregate_options("structure", Mappings.structure_token2int)
        self.predicted_loop = self.aggregate_options("predicted_loop_type", Mappings.pl_token2int)
        self.pair_sequence = self.aggregate_options("pair_sequence", Mappings.sequence_token2int)
        self.ids = self.df["id"].values

        if self.targets is not None:
            target_arr = (
                np.dstack((np.vstack(self.df[col].values) for col in self.targets)).astype(np.float32).clip(-4, 20)
            )
            self.target_values = {seq_id: arr for seq_id, arr in zip(self.ids, target_arr)}

        if self.target_aug:
            errors_arr = (
                np.dstack((np.vstack(self.df[TGT2ERR_COL[col]].values) for col in self.targets))
                .astype(np.float32)
                .clip(-4, 4)
            )
            self.errors_sigma = {seq_id: arr for seq_id, arr in zip(self.ids, errors_arr)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        seq_id = self.ids[idx]
        num_options = len(self.sequence[seq_id])
        option_idx = np.random.choice(list(range(num_options)))
        inputs = {
            "sequence": np.array(self.sequence[seq_id][option_idx]),
            "structure": np.array(self.structure[seq_id][option_idx]),
            "predicted_loop_type": np.array(self.predicted_loop[seq_id][option_idx]),
            "pair_sequence": np.array(self.pair_sequence[seq_id][option_idx]),
        }
        if self.targets is not None:
            targets = self.target_values[seq_id].copy()
        else:
            targets = [0]

        if self.target_aug:
            err_sig = self.errors_sigma[seq_id]
            err = (np.random.gamma(shape=1.2, scale=0.3, size=err_sig.shape) - 0.36) * err_sig
            # err = np.random.randn(*err_sig.shape) * err_sig
            targets[:, : self.num_targets] += err
            targets = targets.clip(-0.5, 100)

        inputs["bpps"] = np.load(f"{self.bpps_path}/{seq_id}.npy").astype("float32")
        return inputs, targets
