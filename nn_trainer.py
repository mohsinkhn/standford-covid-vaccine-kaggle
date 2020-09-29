import argparse
import json
import os

from catalyst import dl
from catalyst.dl import utils
from catalyst.core.callbacks.scheduler import SchedulerCallback
from catalyst.contrib.dl.callbacks.neptune_logger import NeptuneLogger
from catalyst.contrib.nn.schedulers.onecycle import OneCycleLRWithWarmup
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# from torchcontrib.optim import SWA

from constants import FilePaths, TGT_COLS
from datasets import RNAAugData, RNAAugDatav2
from modellib import RNNmodels


def calc_error_mean(row):
    reactivity_error = row['reactivity_error']
    deg_error_Mg_pH10 = row['deg_error_Mg_pH10']
    deg_error_Mg_50C = row['deg_error_Mg_50C']

    return np.mean(np.abs(reactivity_error) +
                   np.abs(deg_error_Mg_pH10) + \
                   np.abs(deg_error_Mg_50C)) / 3


def mcrmse(y_true, y_pred):
    return [mean_squared_error(y_true[:, :, i], y_pred[:, :, i], squared=False) for i in range(y_pred.shape[2])]


class MCRMSE(nn.Module):
    def __init__(self, num_scored=3, eps=1e-8):
        super().__init__()
        self.mse = nn.MSELoss(reduce=None)
        self.num_scored = num_scored
        self.eps = eps

    def forward(self, outputs, targets):
        score = 0
        for idx in range(self.num_scored):
            # weights = targets[:, :, -1]
            # col_mse = torch.sum(weights * self.mse(outputs[:, :, idx], targets[:, :, idx]))/torch.sum(weights)
            col_mse = torch.mean(
                self.mse(outputs[:, :, idx], targets[:, :, idx])
            )
            score += torch.sqrt(col_mse + self.eps) / self.num_scored

        return score


class MCMSRE(nn.Module):
    def __init__(self, num_scored=3, eps=1e-15):
        super().__init__()
        self.mse = nn.MSELoss(reduce=None)
        self.num_scored = num_scored
        self.eps = eps

    def forward(self, outputs, targets):
        score = 0
        for idx in range(self.num_scored):
            out = outputs[:, :, idx]
            targ = targets[:, :, idx]
            out = torch.sign(out) * torch.sqrt(torch.abs(out) + self.eps)
            targ = torch.sign(targ) * torch.sqrt(torch.abs(targ) + self.eps)
            col_mse = torch.mean(
                self.mse(out, targ)
            )
            score += col_mse / self.num_scored

        return score


def train_one_fold(tr, vl, hparams, logger, logdir, device):
    tr_ds = RNAAugDatav2(
        tr,
        targets=TGT_COLS,
        augment_strucures=hparams.get("use_augment", True),
        aug_data_sources=[
                          #"data/augmented_data_public/aug_data5.csv",
                          #"data/augmented_data_public/aug_data5_10.csv",
                          # "data/vienna_7_mec.csv", 
                          "data/vienna_17_mec.csv", 
                          "data/vienna_27_mec.csv",
                          "data/vienna_47_mec.csv",
                           "data/vienna_57_mec.csv",
                            "data/vienna_67_mec.csv"
                           ],
        target_aug=False,
        bpps_path="data/bpps",
    )
    vl_ds = RNAAugDatav2(vl, targets=TGT_COLS, bpps_path="data/bpps")

    tr_dl = DataLoader(tr_ds, shuffle=True, drop_last=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,)
    vl_dl = DataLoader(vl_ds, shuffle=False, drop_last=False, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,)

    model = getattr(RNNmodels, hparams.get("model_name", "RNAGRUModel"))(hparams)
    if hparams.get("optimizer", "adam") == "adam":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=hparams.get("lr", 1e-3), weight_decay=hparams.get("wd", 0), amsgrad=False,
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=hparams.get("lr", 1e-3),
            weight_decay=hparams.get("wd", 0),
            momentum=0.9,
            nesterov=True,
        )
    if hparams.get("scheduler", "reducelrplateau") == "reducelrplateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, factor=0.5, min_lr=2e-4)
    if hparams.get("scheduler", "reducelrplateau") == "one_cycle":
        total_steps = hparams.get("num_epochs") * (len(tr) // hparams.get("batch_size"))
        max_lr = hparams.get("lr", 1e-3)
        scheduler = OneCycleLRWithWarmup(
            optimizer, num_steps=total_steps, lr_range=(max_lr, max_lr / 10, max_lr / 100), warmup_fraction=0.5,
        )
    
    if hparams.get("loss_func", "mcrmse") == "mcrmse":
        criterion = MCRMSE()
    elif hparams.get("loss_func") == "mcmsre":
        criterion = MCMSRE()
    runner = dl.SupervisedRunner(device=device)
    runner.train(
        loaders={"train": tr_dl, "valid": vl_dl},
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=hparams.get("num_epochs", 10),
        logdir=logdir,
        verbose=True,
        callbacks=[logger, SchedulerCallback(mode="epoch")],
        load_best_on_end=True,
        # resume="logs/filter__cnnlstm__posembv5/fold_0/checkpoints/best_full.pth"
    )
    return model, tr_dl, vl_dl


def get_predictions(model, loader, device):
    model.eval()
    model.to(device)
    preds = []
    with torch.no_grad():
        for batch in tqdm(loader):
            x, y = batch
            x = {k: val.to(device) for k, val in x.items()}
            b_preds = model(x)
            b_preds = b_preds.cpu()
            b_preds = b_preds.numpy()
            preds.extend(b_preds)
    return np.array(preds)


def validation_metrics(y_trues, y_preds, sn_flag):
    results = {}
    high_snr_losses = mcrmse(y_trues[sn_flag], y_preds[sn_flag])
    for col, loss in zip(TGT_COLS, high_snr_losses):
        results[f"high_snr_{col}"] = loss

    low_snr_losses = mcrmse(y_trues[~sn_flag], y_preds[~sn_flag])
    for col, loss in zip(TGT_COLS, low_snr_losses):
        results[f"low_snr_{col}"] = loss

    losses = mcrmse(y_trues, y_preds)
    for col, loss in zip(TGT_COLS, losses):
        results[f"{col}"] = loss

    results["total_mcrmse"] = np.mean(losses)
    results["high_snr_mcrmse"] = np.mean(high_snr_losses)
    results["low_snr_mcrmse"] = np.mean(low_snr_losses)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hparams", default="hparams.json")
    parser.add_argument("--tags", required=True)
    args = parser.parse_args()

    with open(args.hparams, "r") as hf:
        hparams = json.load(hf)

    tags = args.tags.split(":")
    exp_dir = Path("./logs") / "__".join(tags)
    exp_dir.mkdir(exist_ok=True, parents=True)

    NUM_WORKERS = 8
    NUM_FOLDS = hparams["num_folds"]

    BATCH_SIZE = hparams.get("batch_size", 32)
    FP = FilePaths("data")
    train = pd.read_json(FP.train_json, lines=True)
    cvlist = list(
        StratifiedKFold(hparams.get("num_folds"), shuffle=True, random_state=hparams.get("seed")).split(
            train, train["SN_filter"]
        )
    )

    with open(str(exp_dir / "hparams.json"), "w") as hf:
        json.dump(hparams, hf)

    run_on_single = hparams.get("run_on_single", False)
    device = utils.get_device()
    val_preds = np.zeros(shape=(len(train), hparams["max_seq_pred"], hparams["num_features"]), dtype="float64",)
    for fold_num in [0, 1, 2, 3, 4]:
        tr_idx, val_idx = cvlist[fold_num]
        tr, vl = train.iloc[tr_idx], train.iloc[val_idx]
        if hparams.get("filter_sn"):
            # tr = tr.loc[tr["signal_to_noise"] > hparams.get("signal_to_noise", 1.0)]
            # vl = vl.loc[vl["signal_to_noise"] > hparams.get("signal_to_noise", 1.0)]
            tr = tr.loc[tr.apply(calc_error_mean, axis=1) < 0.5]
            vl = vl.loc[vl.apply(calc_error_mean, axis=1) < 0.5]
        logdir = exp_dir / f"fold_{fold_num}"
        logdir.mkdir(exist_ok=True)
        neptune_logger = NeptuneLogger(
            api_token=os.environ["NEPTUNE_API_TOKEN"],
            project_name="tezdhar/Covid-RNA-degradation",
            name="covid_rna_degradation",
            params=hparams,
            tags=tags + [f"fold_{fold_num}"],
            upload_source_files=["*.py", "modellib/*.py"],
        )
        trained_model, _, vl_dl = train_one_fold(tr, vl, hparams, neptune_logger, logdir, device)
        vds = RNAAugDatav2(train.iloc[val_idx], targets=TGT_COLS, bpps_path="data/bpps")
        vdl = DataLoader(vds, shuffle=False, drop_last=False, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,)
        val_pred = get_predictions(trained_model, vdl, device)[:, :, : hparams["num_features"]]
        val_preds[val_idx] = val_pred

        if run_on_single:
            break

        if fold_num + 1 != len(cvlist):
            neptune_logger.experiment.stop()

    if run_on_single:
        y_trues = np.dstack((np.vstack(train[col].iloc[val_idx].values) for col in TGT_COLS))
        sn_flag = train["SN_filter"].iloc[val_idx].values.astype(bool)
        eval_results = validation_metrics(y_trues, val_pred, sn_flag)
    else:
        y_trues = np.dstack((np.vstack(train[col].values) for col in TGT_COLS))
        sn_flag = train["SN_filter"].values.astype(bool)
        eval_results = validation_metrics(y_trues, val_preds, sn_flag)

    print(eval_results)
    for eval_name, eval_value in eval_results.items():
        neptune_logger.experiment.log_metric(eval_name, eval_value)
    neptune_logger.experiment.stop()
