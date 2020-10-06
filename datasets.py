"""Datasets required by pytorch dataloaders for competition data."""

from collections import Counter, defaultdict
import json

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from constants import Mappings, FilePaths
import sentencepiece as spm

TGT2ERR_COL = {"reactivity": "reactivity_error", "deg_Mg_50C": "deg_error_Mg_50C", "deg_Mg_pH10": "deg_error_Mg_pH10"}


def get_6gram_tokens(seq, k=5):
    n = len(seq)
    tokens = []
    for i in range(n - k):
        tokens.append(seq[i : i + k])
    start_tokens = [seq[: i * 2 + 1] for i in range(k // 2)]
    end_tokens = [seq[-i * 2 - 1 :] for i in range(k // 2, -1, -1)]

    tokens = start_tokens + tokens + end_tokens
    return tokens


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
    def __init__(
        self, df, targets=None, target_aug=False, augment_strucures=False, aug_data_sources=None, bpps_path="data/bpps"
    ):
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


class RNAAugDatav2(Dataset):
    def __init__(
        self,
        df,
        targets=None,
        target_aug=False,
        augment_strucures=False,
        aug_data_sources=None,
        bpps_path="data/bpps",
        add_segment_info=False,
        add_entropy=False,
        use_6n=False,
    ):
        self.df = df
        self.targets = targets
        self.num_targets = len(targets) if targets is not None else 3
        self.bpps_path = bpps_path
        self.target_aug = target_aug
        self.augment_strucures = augment_strucures
        self.aug_data_sources = aug_data_sources
        self.aug_data = None
        self.data = None
        self.use_6n = use_6n
        self.bpcnt_seg = defaultdict(list)
        self.plcnt_seg = defaultdict(list)
        self.sequence_entropy = defaultdict(list)
        if add_segment_info:
            self.add_segment()
        if add_entropy:
            self.load_entropy()
        if self.augment_strucures:
            self.aug_data = pd.concat([pd.read_csv(src) for src in aug_data_sources])
            in_ids = set(self.df.id.values)
            self.aug_data = self.aug_data.loc[self.aug_data.id.isin(in_ids)]
            print(self.aug_data.head())
        self.prepare_inputs()  # Tokenize stuff and keep as numpy array for quick batch loading

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

    def gather_data(self):
        if self.aug_data is not None:
            self.data = pd.concat([self.df, self.aug_data])
            self.data = self.data.drop_duplicates(subset=["sequence", "structure"])
        else:
            self.data = self.df.copy()

    def tokenize_symbols(self):
        if self.use_6n:
            enc = spm.SentencePieceProcessor(model_file="seq7n.model")
            with open("data/seq7n_word2idx", "r") as fp:
                word2int = json.load(fp)
            self.data["sequence"] = self.data["sequence"].apply(
                lambda x: [word2int.get(token, 0) for token in enc.EncodeAsPieces(" ".join(get_6gram_tokens(x, 5)))]
            )
        else:
            self.data["sequence"] = self.data["sequence"].apply(
                lambda x: [Mappings.sequence_token2int.get(token, 0) for token in x]
            )
        self.data["structure"] = self.data["structure"].apply(
            lambda x: [Mappings.structure_token2int.get(token, 0) for token in x]
        )
        self.data["predicted_loop_type"] = self.data["predicted_loop_type"].apply(
            lambda x: [Mappings.pl_token2int.get(token, 0) for token in x]
        )
        self.data["pair_sequence"] = self.data["pair_sequence"].apply(
            lambda x: [Mappings.sequence_token2int.get(token, 0) for token in x]
        )

    def prepare_inputs(self):
        self.get_match_pairs()
        self.gather_data()
        self.tokenize_symbols()
        self.sequence = self.data.groupby("id")["sequence"].apply(list).to_dict()
        self.structure = self.data.groupby("id")["structure"].apply(list).to_dict()
        self.predicted_loop = self.data.groupby("id")["predicted_loop_type"].apply(list).to_dict()
        self.pair_sequence = self.data.groupby("id")["pair_sequence"].apply(list).to_dict()

        self.ids = self.df["id"].unique()

        if self.targets is not None:
            target_arr = (
                np.dstack((np.vstack(self.df[col].values) for col in self.targets)).astype(np.float32).clip(-4, 20)
            )
            self.target_values = {seq_id: arr for seq_id, arr in zip(self.df.id.values, target_arr)}

        if self.target_aug:
            errors_arr = (
                np.dstack((np.vstack(self.df[TGT2ERR_COL[col]].values) for col in self.targets))
                .astype(np.float32)
                .clip(-4, 4)
            )
            self.errors_sigma = {seq_id: arr for seq_id, arr in zip(self.df.id.values, errors_arr)}

    def get_bpps(self, seq_id):
        b1 = [np.load(f"{self.bpps_path}/{seq_id}.npy").astype("float32")]
        b2 = [np.load(f"data/vienna_2/bpps/{seq_id}_{T}.npy").astype("float32") for T in [7, 17, 27, 47, 57, 67]]
        b3 = [np.load(f"data/nupack_95/bpps/{seq_id}_{T}.npy").astype("float32") for T in [7, 17, 27, 37, 47, 57, 67]]
        b4 = [np.load(f"data/nupack_99/bpps/{seq_id}_{T}.npy").astype("float32") for T in [37]]
        bpps = b1 + b2 + b3 + b4
        return np.dstack(bpps)

    def _map_segment_counts(self, x):
        c = Counter(x)
        return [c[sym] for sym in x]

    def _split_seg(self, x):
        x = x.split("_")
        return x

    def _bpseg2int(self, x):
        return [int(sym) if sym != "-1" else -1 for sym in x]

    def _plseg2int(self, x):
        return [Mappings.pl2_token2int[sym] for sym in x]

    def add_segment(self):
        sdf = pd.read_csv("data/predicted_loop_segments.csv")
        sdf = pd.merge(self.df, sdf, on="id", how="left")
        sdf["bp_seg_num"] = sdf["bp_seg_num"].apply(self._split_seg)
        sdf["bp_seg_num"] = sdf["bp_seg_num"].apply(self._bpseg2int)
        sdf["bp_seg_count"] = sdf["bp_seg_num"].apply(self._map_segment_counts)
        sdf["pl_seg_num"] = sdf["pl_seg_num"].apply(self._split_seg)
        sdf["pl_seg_num"] = sdf["pl_seg_num"].apply(self._plseg2int)
        sdf["pl_seg_count"] = sdf["pl_seg_num"].apply(self._map_segment_counts)
        bpmax_seg = 9
        plmax_seg = 77

        self.bpcnt_seg = defaultdict(list)
        self.plcnt_seg = defaultdict(list)
        for row_idx, row in sdf[["id", "bp_seg_num", "bp_seg_count", "pl_seg_num", "pl_seg_count"]].iterrows():
            bp_num = row["bp_seg_num"]
            bp_cnts = row["bp_seg_count"]
            pl_num = row["pl_seg_num"]
            pl_cnts = row["pl_seg_count"]

            n = len(row["bp_seg_num"])
            bpcnt_array = np.zeros((n, bpmax_seg))
            plcnt_array = np.zeros((n, plmax_seg))
            for j in range(n):
                bpcnt_array[j, bp_num[j]] = bp_cnts[j] / 40
                plcnt_array[j, pl_num[j]] = pl_cnts[j] / 40
            self.bpcnt_seg[row["id"]] = bpcnt_array.astype("float32")
            self.plcnt_seg[row["id"]] = plcnt_array.astype("float32")

    def get_embeddings(self, seq_id):
        return np.load(f"data/w2v_embeddings/{seq_id}.npy")

    def load_entropy(self):
        with open("data/sequence_entropy.json", "r") as fp:
            self.sequence_entropy = json.load(fp)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        seq_id = self.ids[idx]
        num_options = len(self.sequence[seq_id])
        option_idx = np.random.choice(list(range(num_options)))
        inputs = {
            "sequence": np.array(self.sequence[seq_id][option_idx]),
            "structure": np.array(self.structure[seq_id][option_idx]),
            "predicted_loop_type": np.array(self.predicted_loop[seq_id][option_idx]),
            "pair_sequence": np.array(self.pair_sequence[seq_id][option_idx]),
            "sequence_bp_segment": self.bpcnt_seg.get(seq_id, np.ones(shape=(10, 2))),
            "sequence_pl_segment": self.plcnt_seg.get(seq_id, np.ones(shape=(10, 2))),
            "sequence_entropy": np.array(self.sequence_entropy.get(seq_id)).astype("float32"),
            # "pair_index": np.array(self.pair_sequence[seq_id][option_idx])
        }
        if self.targets is not None:
            targets = self.target_values[seq_id].copy()
        else:
            targets = [0]

        if self.target_aug:
            err_sig = self.errors_sigma[seq_id]
            err = np.random.normal(loc=0.0, scale=0.001, size=err_sig.shape)
            targets[:, : self.num_targets] += err

        inputs["bpps"] = self.get_bpps(seq_id)
        return inputs, targets


class BPPSData(Dataset):
    def __init__(self, df, bpps_path="data/bpps"):
        self.df = df.copy()
        self.targets = None
        self.bpps_path = bpps_path
        self.prepare_inputs()  # Tokenize stuff and keep as numpy array for quick batch loading

    @staticmethod
    def base_pair_index_to_sequence(df):  # From base pair index, get actual index
        df["pair_sequence"] = df[["sequence", "pair_index"]].apply(
            lambda x: [x["sequence"][idx] if idx >= 0 else "0" for idx in x["pair_index"]], axis=1
        )
        return df

    def get_match_pairs(self):  # Get index of base pairs
        self.df["pair_index"] = self.df["structure"].apply(lambda x: match_pair(x))
        self.df = self.base_pair_index_to_sequence(self.df)

    def tokenize_symbols(self):
        self.df["sequence"] = self.df["sequence"].apply(
            lambda x: [Mappings.sequence_token2int.get(token, 0) for token in x]
        )
        self.df["predicted_loop_type"] = self.df["predicted_loop_type"].apply(
            lambda x: [Mappings.pl_token2int.get(token, 0) for token in x]
        )
        self.df["structure"] = self.df["structure"].apply(
            lambda x: [Mappings.structure_token2int.get(token, 0) for token in x]
        )
        self.df["pair_sequence"] = self.df["pair_sequence"].apply(
            lambda x: [Mappings.sequence_token2int.get(token, 0) for token in x]
        )

    def prepare_inputs(self):
        self.get_match_pairs()
        self.tokenize_symbols()
        self.sequence = self.df["sequence"].apply(list).values
        self.predicted_loop = self.df["predicted_loop_type"].apply(list).values
        self.structure = self.df["structure"].apply(list).values
        self.pair_sequence = self.df["pair_sequence"].apply(list).values
        self.ids = self.df["id"].unique()
        self.targets = self.df.id.apply(lambda x: self.get_bpps(x)).values

    def get_bpps(self, seq_id):
        b1 = np.load(f"{self.bpps_path}/{seq_id}.npy").astype("float32")
        return b1.sum(1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        inputs = {
            "sequence": np.array(self.sequence[idx]),
            "structure": np.array(self.structure[idx]),
            "predicted_loop_type": np.array(self.predicted_loop[idx]),
            "pair_sequence": np.array(self.pair_sequence[idx])
        }
        return inputs, inputs  # np.array(self.targets[idx]).reshape(-1, 1)
