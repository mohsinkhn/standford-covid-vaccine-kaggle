import argparse
import json
import os

from catalyst import dl
from catalyst.dl import utils
from catalyst.contrib.dl.callbacks.neptune_logger import NeptuneLogger
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
import torch
from torch import nn
from torch.utils.data import DataLoader

# from torchcontrib.optim import SWA

from constants import FilePaths, TGT_COLS
from datasets import RNAData
from modellib.RNNmodels import RNAGRUModel


def mcrmse(y_true, y_pred):
    return [mean_squared_error(y_true[:, :, i], y_pred[:, :, i], squared=False) for i in range(y_pred.shape[2])]


class MCRMSE(nn.Module):
    def __init__(self, num_scored=3, eps=1e-8):
        super().__init__()
        self.mse = nn.MSELoss()
        self.num_scored = num_scored
        self.eps = eps

    def forward(self, outputs, targets):
        score = 0
        for idx in range(self.num_scored):
            score += torch.sqrt(self.mse(outputs[:, :, idx], targets[:, :, idx]) + self.eps) / self.num_scored

        return score


def train_one_fold(tr, vl, hparams, logger, logdir, device):
    tr_ds = RNAData(tr, targets=TGT_COLS)
    vl_ds = RNAData(vl, targets=TGT_COLS)

    tr_dl = DataLoader(tr_ds, shuffle=True, drop_last=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    vl_dl = DataLoader(vl_ds, shuffle=False, drop_last=False, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    model = RNAGRUModel(hparams)
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams.get("lr", 1e-3), weight_decay=hparams.get("wd", 0))
    # optimizer = SWA(base_opt, swa_start=10, swa_freq=5, swa_lr=hparams.get("lr", 1e-2))

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)
    criterion = MCRMSE()
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
        callbacks=[logger],
        load_best_on_end=True,
    )
    return model, tr_dl, vl_dl


def get_predictions(model, loader, device):
    model.eval()
    model.to(device)
    preds = []
    with torch.no_grad():
        for batch in loader:
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

    device = utils.get_device()
    val_preds = np.zeros(shape=(len(train), hparams["max_seq_pred"], hparams["num_features"]), dtype='float64')
    for fold_num, (tr_idx, val_idx) in enumerate(cvlist):
        tr, vl = train.iloc[tr_idx], train.iloc[val_idx]
        if hparams.get("filter_sn"):
            tr = tr.loc[tr["SN_filter"] == 1]
            vl = vl.loc[vl["SN_filter"] == 1]
        logdir = exp_dir / f"fold_{fold_num}"
        logdir.mkdir(exist_ok=True)
        neptune_logger = NeptuneLogger(
            api_token=os.environ["NEPTUNE_API_TOKEN"],
            project_name="tezdhar/Covid-RNA-degradation",
            name="covid_rna_degradation",
            params=hparams,
            tags=tags+[f"fold_{fold_num}"],
            upload_source_files=["*.py", "modellib/*.py"],
        )
        trained_model, _, vl_dl = train_one_fold(tr, vl, hparams, neptune_logger, logdir, device)
        vds = RNAData(train.iloc[val_idx], targets=TGT_COLS)
        vdl = DataLoader(vds, shuffle=False, drop_last=False, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
        val_preds[val_idx] = get_predictions(trained_model, vdl, device)[:, :, : hparams["num_features"]]

        if fold_num + 1 != len(cvlist):
            neptune_logger.experiment.stop()

    y_trues = np.dstack((np.vstack(train[col].values) for col in TGT_COLS))
    sn_flag = train["SN_filter"].values.astype(bool)
    eval_results = validation_metrics(y_trues, val_preds, sn_flag)
    for eval_name, eval_value in eval_results.items():
        neptune_logger.experiment.log_metric(eval_name, eval_value)
    neptune_logger.experiment.stop()
