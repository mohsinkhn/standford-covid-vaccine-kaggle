import numpy as np
from pathlib import Path
import pandas as pd
import torch
from torch_geometric.data import Data

from constants import TGT_COLS, FilePaths


def sample_is_clean(row):
    return row['SN_filter'] == 1


def calc_error_mean(row):
    reactivity_error = row['reactivity_error']
    deg_error_Mg_pH10 = row['deg_error_Mg_pH10']
    deg_error_Mg_50C = row['deg_error_Mg_50C']

    return np.mean(np.abs(reactivity_error) +
                   np.abs(deg_error_Mg_pH10) +
                   np.abs(deg_error_Mg_50C)) / 3


def calc_sample_weight(row, threshold):
    if sample_is_clean(row):
        return 1.
    else:
        error_mean = calc_error_mean(row)
        if error_mean >= threshold:
            return 0.

        return 1. - error_mean / threshold


class MyData(Data):
    def __init__(
        self, x=None, edge_index=None, edge_attr=None, y=None, pos=None, norm=None, face=None, weight=None, **kwargs
    ):
        super(MyData, self).__init__(
            x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, pos=pos, norm=norm, face=face, **kwargs
        )
        self.weight = weight
    
    def __len__(self):
        return len(self.x)


def get_pairs(struc):
    n = len(struc)
    stack1 = []
    pairs = []
    for i in range(n):
        s = struc[i]
        if s == "(":
            stack1.append(i)
        elif s == ")":
            pairs.append((stack1.pop(), i))
    return pairs


def node_feature_vec(sequence, struc, predicted_loop_type, bpps_row):
    return [
        0,  # bp segment node
        0,  # pl segment node
        sequence == "A",  # seq
        sequence == "C",  # seq
        sequence == "G",  # seq
        sequence == "U",  # seq
        predicted_loop_type == "S",
        predicted_loop_type == "M",
        predicted_loop_type == "I",
        predicted_loop_type == "B",
        predicted_loop_type == "H",
        predicted_loop_type == "E",
        predicted_loop_type == "X",
        sum(bpps_row),
        max(bpps_row),
        (struc == "(") or (struc == ")"),  # is paired
        0,  # seg pairs
    ]


def ploop_node_feature_vec(predicted_loop_type, pcnt):
    return [
        0,  # bp segment node
        pcnt,  # pl segment node
        0,  # seq
        0,  # seq
        0,  # seq
        0,  # seq
        predicted_loop_type == "S",
        predicted_loop_type == "M",
        predicted_loop_type == "I",
        predicted_loop_type == "B",
        predicted_loop_type == "H",
        predicted_loop_type == "E",
        predicted_loop_type == "X",
        0,
        0,
        predicted_loop_type == "S",  # is paired
        0,  # seg pairs
    ]


def pair_node_feature_vec(num_pairs):
    return [
        1,  # bp segment node
        0,  # pl segment node
        0,  # seq
        0,  # seq
        0,  # seq
        0,  # seq
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,  # is paired
        num_pairs,  # seg pairs
    ]


def get_node_features(sequence, structure, predicted_loop_type, bpps, segment_bp_num, segment_bp_pairs, segment_pl_num):
    node_features = []
    for (seq, struc, pl, bpps_row) in zip(sequence, structure, predicted_loop_type, bpps):
        node_features.append(node_feature_vec(seq, struc, pl, bpps_row))

    segment_bp, seg_idx = np.unique(segment_bp_num.split("_"), return_index=True)
    segment_pairs = [int(c) for c in segment_bp_pairs.split("_")]
    for bps, bpidx in zip(segment_bp, seg_idx):
        if bps == "-1":
            continue
        node_features.append(pair_node_feature_vec(segment_pairs[bpidx] * 2 / len(sequence)))

    segment_pl, pl_idx, pl_cnts = np.unique(segment_pl_num.split("_"), return_index=True, return_counts=True)
    for plseg, plidx, plcnt in zip(segment_pl, pl_idx, pl_cnts):
        if plseg == "0":
            continue
        node_features.append(ploop_node_feature_vec(predicted_loop_type[plidx], plcnt/10))

    return node_features


def add_edges(edge_index, edge_features, node1, node2, feature1, feature2):
    edge_index.append([node1, node2])
    edge_features.append(feature1)
    edge_index.append([node2, node1])
    edge_features.append(feature2)


def add_edges_forward(edge_index, edge_features, node1, node2, feature1, feature2):
    edge_index.append([node1, node2])
    edge_features.append(feature1)


def add_edges_between_base_nodes(edge_index, edge_features, node1, node2, bpps_np1, bpps_np2):
    edge_feature1 = [
        0,  # is edge for paired nodes
        0,  # is edge between bp node and non bp node
        0,  # is edge between bp node and non bp node
        0,  # pl node
        0,
        1,  # forward edge: 1, backward edge: -1
        0,  # forward or backward
        0,  # bpps if edge is for paired nodes
        bpps_np1,  # non pairing prob
    ]
    edge_feature2 = [
        0,  # is edge for paired nodes
        0,  # is edge between bp node and non bp node
        0,  # is edge between bp node and non bp node
        0,  # pl node
        0,
        -1,  # forward edge: 1, backward edge: -1
        0,  # forward or backward
        0,  # bpps if edge is for paired nodes
        bpps_np2,  # non pairing prob
    ]
    add_edges(edge_index, edge_features, node1, node2, edge_feature1, edge_feature2)
    #add_edges_forward(edge_index, edge_features, node1, node2, edge_feature1, edge_feature2)


def add_edges_between_paired_nodes(edge_index, edge_features, node1, node2, bpps, bpps_np1, bpps_np2):
    edge_feature1 = [
        1,  # is edge for paired nodes
        0,  # is edge between bp node and non bp node
        0,  # is edge between bp node and non bp node
        0,  # pl node
        0,
        0,  # forward edge: 1, backward edge: -1
        0,  # forward or backward
        (np.log(bpps + 1e-6).clip(-10, 10) + 10)/20,  # bpps if edge is for paired nodes
        bpps_np1
    ]
    edge_feature2 = [
        1,  # is edge for paired nodes
        0,  # is edge between bp node and non bp node
        0,  # is edge between bp node and non bp node
        0,  # pl node
        0,
        0,  # forward edge: 1, backward edge: -1
        0,  # forward or backward
        (np.log(bpps + 1e-6).clip(-10, 10) + 10)/20,  # bpps if edge is for paired nodes
        bpps_np2
    ]
    add_edges(edge_index, edge_features, node1, node2, edge_feature1, edge_feature2)


def add_edges_between_bp_segment(edge_index, edge_features, node1, node2, bpps_sum):
    edge_feature1 = [
        0,  # is edge for paired nodes
        1,  # is edge between bp node and non bp node
        0,  # is edge between bp node and non bp node
        0,  # pl - pl edges
        0,  # pl - segment edges
        0,  # forward edge: 1, backward edge: -1
        0,  # forward or backward
        1,  # bpps if edge is for paired nodes
        0,  # non pairing prob
    ]
    edge_feature2 = [
        0,  # is edge for paired nodes
        1,  # is edge between bp node and non bp node
        0,  # is edge between bp node and non bp node
        0,  # pl - pl node
        0,  # pl - segment
        0,  # forward edge: 1, backward edge: -1
        0,  # forward or backward
        1,  # bpps if edge is for paired nodes
        0  # bpps non pairing prob
    ]
    add_edges(edge_index, edge_features, node1, node2, edge_feature1, edge_feature2)


def add_edges_between_pl_nodes(edge_index, edge_features, node1, node2):
    edge_feature1 = [
        0,  # is edge for paired nodes
        0,  # is edge between bp node and non bp node
        0,  # is edge between bp node and non bp node
        0,  # pl node
        1,
        0,  # forward edge: 1, backward edge: -1
        0,  # forward or backward
        0,  # bpps if edge is for paired nodes
        0
    ]
    edge_feature2 = [
        0,  # is edge for paired nodes
        0,  # is edge between bp node and non bp node
        0,  # is edge between bp node and non bp node
        0,  # pl node
        1,
        0,  # forward edge: 1, backward edge: -1
        0,  # forward or backward
        0,  # bpps if edge is for paired nodes
        0
    ]
    add_edges(edge_index, edge_features, node1, node2, edge_feature1, edge_feature2)


def add_edges_between_pl_segments(edge_index, edge_features, node1, node2):
    edge_feature1 = [
        0,  # is edge for paired nodes
        0,  # is edge between bp node and non bp node
        0,  # is edge between bp node and non bp node
        1,  # pl node
        0,
        0,  # forward edge: 1, backward edge: -1
        0,  # forward or backward
        0,  # bpps if edge is for paired nodes
        0
    ]
    edge_feature2 = [
        0,  # is edge for paired nodes
        0,  # is edge between bp node and non bp node
        0,  # is edge between bp node and non bp node
        1,  # pl node
        0,
        0,  # forward edge: 1, backward edge: -1
        0,  # forward or backward
        0,  # bpps if edge is for paired nodes
        0
    ]
    add_edges(edge_index, edge_features, node1, node2, edge_feature1, edge_feature2)


def get_segment_pairs(segment):
    segs, seg_idx, seg_rev_idx = np.unique(segment, return_index=True, return_inverse=True)
    seg_pairs = []
    for i, seg in enumerate(segment[:-1]):
        if (seg == segment[i + 1]) or (seg == "0") or (segment[i + 1] == "0"):
            continue
        if segs[0] == "0":
            pair = (seg_rev_idx[i] - 1, seg_rev_idx[i + 1] - 1)
        else:
            pair = (seg_rev_idx[i], seg_rev_idx[i + 1])
        rev_pair = (pair[1], pair[0])
        if (rev_pair not in seg_pairs) and (pair not in seg_pairs):
            seg_pairs.append(pair)
    return seg_pairs


def get_edge_index_features(
    sequence, structure, predicted_loop_type, bpps, segment_bp_num, segment_bp_pairs, segment_pl_num,
    add_bp_master=True, add_pl_master=True, add_pl_pl=True
):
    edge_index = []
    edge_features = []
    for j in range(len(sequence) - 1):
        add_edges_between_base_nodes(edge_index, edge_features, j, j + 1, 1 - sum(bpps[j]), 1 - sum(bpps[j+1]))

    pairs = get_pairs(structure)
    for pp in pairs:
        i, j = pp
        add_edges_between_paired_nodes(edge_index, edge_features, i, j, bpps[i, j], 1 - sum(bpps[i]), 1 - sum(bpps[j]))

    segment_bp, seg_idx = np.unique(segment_bp_num.split("_"), return_inverse=True)
    node_cnt = len(sequence)
    seq_num = np.array(range(len(sequence)))
    bpnodes = 0
    for bpidx, bps in enumerate(segment_bp):
        if bps == "-1":
            continue
        base_nodes = seq_num[seg_idx == bpidx]
        for node in base_nodes:
            if add_bp_master:
                add_edges_between_bp_segment(edge_index, edge_features, node_cnt + bpnodes, node, sum(bpps[node]))
        bpnodes += 1

    node_cnt = len(sequence) + bpnodes
    pl_nodes = 0
    segment_pl, pl_idx = np.unique(segment_pl_num.split("_"), return_inverse=True)
    seg_pairs = get_segment_pairs(segment_pl)

    for plidx, plseg in enumerate(segment_pl):
        if plseg == "0":
            continue
        base_nodes = seq_num[pl_idx == plidx]
        for node in base_nodes:
            if add_pl_master:
                add_edges_between_pl_segments(edge_index, edge_features, node_cnt + pl_nodes, node)
        pl_nodes += 1

    for node1, node2 in seg_pairs:
        if add_pl_pl:
            add_edges_between_pl_nodes(edge_index, edge_features, node_cnt + node1, node_cnt + node2)

    return edge_index, edge_features


def preprare_graph(
    sequence,
    structure,
    predicted_loop_type,
    bpps,
    segment_bp,
    segment_bp_pairs,
    segment_pl,
    targets=None,
    weight=None,
    seq_scored=68,
    seq_len=107,
    add_bp_master=True,
    add_pl_master=True,
    add_pl_pl=True
):
    node_features = get_node_features(
        sequence, structure, predicted_loop_type, bpps, segment_bp, segment_bp_pairs, segment_pl
    )
    num_nodes = len(node_features)
    train_mask = torch.tensor([True] * seq_scored + [False] * (num_nodes - seq_scored))
    test_mask = torch.tensor([True] * seq_len + [False] * (num_nodes - seq_len))
    edge_index, edge_features = get_edge_index_features(
        sequence, structure, predicted_loop_type, bpps, segment_bp, segment_bp_pairs, segment_pl,
        add_bp_master=add_bp_master, add_pl_master=add_pl_master, add_pl_pl=add_pl_pl
    )
    node_features = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_features = torch.tensor(edge_features, dtype=torch.float)

    if targets is not None:
        targets_all = []
        for nn in range(num_nodes):
            if nn < seq_scored:
                targets_all.append([targets[0][nn], targets[1][nn], targets[2][nn]])
            else:
                targets_all.append([0, 0, 0])
        targets = torch.tensor(targets_all, dtype=torch.float)

    if weight is not None:
        weights = [weight] * seq_scored + [0] * (num_nodes - seq_scored)
        weight = torch.tensor(weights, dtype=torch.float)

    return MyData(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_features,
        y=targets,
        weight=weight,
        train_mask=train_mask,
        test_mask=test_mask,
    )


class GraphDataset(object):
    def __init__(self, df, is_train, hparams,
                 segment_csv_path="data/predicted_loop_segments.csv", bpps_paths=["data/bpps"]):
        
        self.is_train = is_train
        self.hparams = hparams
        self.bpps_paths = bpps_paths
        merged_df = self.get_concat_df(df, segment_csv_path)
        self.data = self.prepare_dataset(merged_df)
    
    def get_concat_df(self, df, segment_csv_path):
        df_segments = pd.read_csv(segment_csv_path)
        merged_df = pd.merge(df, df_segments, on="id", how="left")
        return merged_df

    def __len__(self):
        return len(self.data)

    def prepare_dataset(self, df):
        data = []
        for row_idx, row in df.iterrows():
            targets = None
            weight = None
            if self.is_train:
                targets = [row[col] for col in TGT_COLS]
                # weight = 1.0  # calc_sample_weight(row, threshold)
            bpps = np.mean([np.load(str(Path(path) / f"{row['id']}.npy")) for path in self.bpps_paths], 0)
            data_ = preprare_graph(
                sequence=row["sequence"],
                structure=row["structure"],
                predicted_loop_type=row["pstructure"],
                bpps=bpps,
                segment_bp=row["bp_seg_num"],
                segment_bp_pairs=row["bp_seg_pairs"],
                segment_pl=row["pl_seg_num"],
                targets=targets,
                weight=weight,
                seq_scored=row["seq_scored"],
                seq_len=row["seq_length"],
                add_bp_master=self.hparams["add_bp_master"],
                add_pl_master=self.hparams["add_pl_master"],
                add_pl_pl=self.hparams["add_pl_pl"]
            )
            data.append(data_)
        return data


# def prepare_dataset(df, is_train=True, threshold=0.5, add_bp_master=True, add_pl_master=True, add_pl_pl=True):
#     df_segments = pd.read_csv("data/predicted_loop_segments.csv")
#     df = pd.merge(df, df_segments, on="id", how="left")
#     data = []
#     for row_idx, row in df.iterrows():
#         targets = None
#         weight = None
#         if is_train:
#             targets = [row["reactivity"], row["deg_Mg_50C"], row["deg_Mg_pH10"]]
#             weight = 1.0  # calc_sample_weight(row, threshold)
#         bpps = np.load(f"data/bpps/{row['id']}.npy")
#         # bpps = np.log(bpps + 1e-8).clip(-10, 10) / 5
#         data_ = preprare_graph(
#             sequence=row["sequence"],
#             structure=row["structure"],
#             predicted_loop_type=row["pstructure"],
#             bpps=bpps,
#             segment_bp=row["bp_seg_num"],
#             segment_bp_pairs=row["bp_seg_pairs"],
#             segment_pl=row["pl_seg_num"],
#             targets=targets,
#             weight=weight,
#             seq_scored=row["seq_scored"],
#             seq_len=row["seq_length"],
#             add_bp_master=add_bp_master,
#             add_pl_master=add_pl_master,
#             add_pl_pl=add_pl_pl
#         )
#         data.append(data_)
#     return data
