import os
import numpy as np
import pandas as pd
import random
import torch
from torch.nn import Linear, LayerNorm, ReLU, Dropout
from torch_geometric.nn import ChebConv, NNConv, DeepGCNLayer, EdgeConv
from torch_geometric.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import copy
from catalyst.dl import utils

from constants import FilePaths
from modellib.graphmodel import MyDeeperGCN
from pytorch_geometric_dataset import calc_error_mean, prepare_dataset


def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def weighted_mse_loss(prds, tgts, weight):
    return torch.mean(weight * (prds - tgts) ** 2)


def criterion(prds, tgts, weight=None):
    if weight is None:
        return (
            torch.sqrt(torch.nn.MSELoss()(prds[:, 0], tgts[:, 0]))
            + torch.sqrt(torch.nn.MSELoss()(prds[:, 1], tgts[:, 1]))
            + torch.sqrt(torch.nn.MSELoss()(prds[:, 2], tgts[:, 2]))
        ) / 3
    else:
        return (
            torch.sqrt(weighted_mse_loss(prds[:, 0], tgts[:, 0], weight))
            + torch.sqrt(weighted_mse_loss(prds[:, 1], tgts[:, 1], weight))
            + torch.sqrt(weighted_mse_loss(prds[:, 2], tgts[:, 2], weight))
        ) / 3


def build_id_seqpos(df):
    id_seqpos = []
    for i in range(len(df)):
        id = df.loc[i, "id"]
        seq_length = df.loc[i, "seq_length"]
        for seqpos in range(seq_length):
            id_seqpos.append(id + "_" + str(seqpos))
    return id_seqpos


def sample_is_clean(row):
    return row["SN_filter"] == 1
    # return row['signal_to_noise'] > 1 and \
    #       min((min(row['reactivity']),
    #            min(row['deg_Mg_pH10']),
    #            min(row['deg_pH10']),
    #            min(row['deg_Mg_50C']),
    #            min(row['deg_50C']))) > -0.5


# categorical value for target (used for stratified kfold)
def add_y_cat(df):
    target_mean = df["reactivity"].apply(np.mean) + df["deg_Mg_pH10"].apply(np.mean) + df["deg_Mg_50C"].apply(np.mean)
    df["y_cat"] = pd.qcut(np.array(target_mean), q=20).codes


def get_dataloader(df, hparams, shuffle=True):
    # data_train = build_data(df.reset_index(drop=True), True)
    data_train = prepare_dataset(df, True, 0.5, add_bp_master=hparams["add_bp_master"],
     add_pl_master=hparams["add_pl_master"], add_pl_pl=hparams["add_pl_pl"])
    return data_train, DataLoader(data_train, batch_size=hparams["batch_size"], shuffle=shuffle, num_workers=8)


def train_fold(model, loader_train, loader_valid, optimizer, criterion, epochs, device):
    best_mcrmse = np.inf
    for epoch in range(epochs):
        print("Epoch", epoch)
        model.train()
        train_loss = 0.0
        nb = 0
        for data in tqdm(loader_train):
            data = data.to(device)
            mask = data.train_mask
            weight = data.weight[mask]

            optimizer.zero_grad()
            out = model(data)[mask]
            y = data.y[mask]
            loss = criterion(out, y, weight)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * y.size(0)
            nb += y.size(0)

            del data
            del out
            del y
            del loss
            # gc.collect()
            # torch.cuda.empty_cache()
        train_loss /= nb

        model.eval()
        valid_loss = 0.0
        nb = 0
        ys = torch.zeros((0, 3)).to(device).detach()
        outs = torch.zeros((0, 3)).to(device).detach()
        for data in tqdm(loader_valid):
            data = data.to(device)
            mask = data.train_mask

            out = model(data)[mask].detach()
            y = data.y[mask].detach()
            loss = criterion(out, y).detach()
            valid_loss += loss.item() * y.size(0)
            nb += y.size(0)

            outs = torch.cat((outs, out), dim=0)
            ys = torch.cat((ys, y), dim=0)

            del data
            del out
            del y
            del loss
            # gc.collect()
            # torch.cuda.empty_cache()
        valid_loss /= nb

        mcrmse = criterion(outs, ys).item()

        print("T Loss: {:.4f} V Loss: {:.4f} V MCRMSE: {:.4f}".format(train_loss, valid_loss, mcrmse))

        if mcrmse < best_mcrmse:
            print("Best valid MCRMSE updated to", mcrmse)
            best_mcrmse = mcrmse
            best_model_state = copy.deepcopy(model.state_dict())
    return best_model_state


if __name__ == "__main__":
    seed = 122345786
    set_seed(seed)

    FN = FilePaths("data")
    HPARAMS = {
        "nb_fold": 5,
        "filter_noise": True,
        "signal_to_noise_ratio": 0.5,
        "num_epochs": 100,
        "batch_size": 8,
        "lr": 3e-4,
        "wd": 0,
        "num_layers": 10,
        "node_hidden_channels": 128,
        "edge_hidden_channels": 32,
        "dropout1": 0.1,
        "dropout2": 0.1,
        "dropout3": 0.1,
        "hidden_channels3": 64,
        "T": 2,
        "add_bp_master": True,
        "add_pl_master": True,
        "add_pl_pl": True
    }
    df_tr = pd.read_json(FN.train_json, lines=True)
    add_y_cat(df_tr)

    device = utils.get_device()
    all_ys = torch.zeros((0, 3)).to(device).detach()
    all_outs = torch.zeros((0, 3)).to(device).detach()
    best_model_states = []
    cvlist = list(StratifiedKFold(HPARAMS["nb_fold"], shuffle=True, random_state=seed).split(df_tr, df_tr["y_cat"]))

    for fold, (tr_idx, vl_idx) in enumerate(cvlist):
        tr, vl = df_tr.iloc[tr_idx], df_tr.iloc[vl_idx]

        if HPARAMS["filter_noise"]:
            cond = tr.apply(calc_error_mean, axis=1) < 0.5
            tr = tr.loc[cond].reset_index(drop=True)

        vl = vl.loc[vl["SN_filter"] == 1].reset_index(drop=True)
        print(tr.shape, vl.shape)

        data_train, loader_train = get_dataloader(tr, HPARAMS, shuffle=True)
        data_valid, loader_valid = get_dataloader(vl, HPARAMS, shuffle=False)

        model = MyDeeperGCN(
            data_train[0].num_node_features,
            data_train[0].num_edge_features,
            node_hidden_channels=HPARAMS["node_hidden_channels"],
            edge_hidden_channels=HPARAMS["edge_hidden_channels"],
            num_layers=HPARAMS["num_layers"],
            dropout1=HPARAMS["dropout1"],
            dropout2=HPARAMS["dropout2"],
            dropout3=HPARAMS["dropout3"],
            T=HPARAMS["T"],
            hidden_channels3=HPARAMS["hidden_channels3"],
            num_classes=3,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=HPARAMS["lr"], weight_decay=HPARAMS["wd"])
        best_state = train_fold(model, loader_train, loader_valid, optimizer, criterion, HPARAMS["num_epochs"], device)
        best_model_states.append(best_state)
        torch.save(model.state_dict(), f"logs/gcn_model_run2/model_state_{fold}.pt")

        # break

        model.load_state_dict(best_state)
        model.eval()
        for data in tqdm(loader_valid):
            data = data.to(device)
            mask = data.train_mask

            out = model(data)[mask].detach()
            y = data.y[mask].detach()

            all_ys = torch.cat((all_ys, y), dim=0)
            all_outs = torch.cat((all_outs, out), dim=0)

            del data
            del out
            del y

    print('CV MCRMSE ', criterion(all_outs, all_ys).item())
    del all_outs
    del all_ys

