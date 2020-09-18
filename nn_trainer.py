import argparse
import json
import os

from catalyst import dl
from catalyst.contrib.dl.callbacks.neptune_logger import NeptuneLogger
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
import torch
from torch import nn
from torch.utils.data import DataLoader
# from torchcontrib.optim import SWA

from constants import TRAIN_JSON
from datasets import RNAData
from modellib.RNNmodels import RNAGRUModel


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


def train_one_fold(tr, vl, hparams, logger, logdir):
    tr_ds = RNAData(tr, targets=["reactivity", "deg_Mg_pH10", "deg_Mg_50C"])
    vl_ds = RNAData(vl, targets=["reactivity", "deg_Mg_pH10", "deg_Mg_50C"])

    tr_dl = DataLoader(tr_ds, shuffle=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    vl_dl = DataLoader(vl_ds, shuffle=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    model = RNAGRUModel(hparams)
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams.get("lr", 1e-3), weight_decay=hparams.get("wd", 0))
    # optimizer = SWA(base_opt, swa_start=10, swa_freq=5, swa_lr=hparams.get("lr", 1e-2))

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)
    criterion = MCRMSE()
    runner = dl.SupervisedRunner()
    runner.train(
        loaders={"train": tr_dl, "valid": vl_dl},
        model=model, criterion=criterion, optimizer=optimizer, scheduler=scheduler,
        num_epochs=100, logdir=logdir, verbose=True,
        callbacks=[logger],
        load_best_on_end=True,
    )
    return model


def get_predictions(model, loader, cuda=False):
    model.eval()
    if cuda:
        model.cuda()
    preds = []
    with torch.no_grad():
        for batch in loader:
            x, _ = batch
            if cuda:
                x = x.cuda()
                b_preds = model(x)
                b_preds = b_preds.cpu()
            else:
                b_preds = model(x)
            b_preds = b_preds.numpy()
            preds.extend(b_preds)
    return np.array(preds)


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
    BATCH_SIZE = hparams.get("batch_size", 32)

    train = pd.read_json(TRAIN_JSON, lines=True)
    train = train.loc[train["SN_filter"] == 1]
    cvlist = list(StratifiedKFold(hparams.get("num_folds", 10), shuffle=True, random_state=hparams.get("seed", 978654)).split(train, train["SN_filter"]))

    neptune_logger = NeptuneLogger(
                    api_token=os.environ["NEPTUNE_API_TOKEN"],
                    project_name='tezdhar/Covid-RNA-degradation',
                    name='covid_rna_degradation',
                    params=hparams,
                    tags=tags,
                    upload_source_files=['*.py', 'modellib/*.py'],
                    )

    for fold_num, (tr_idx, val_idx) in enumerate(cvlist):
        tr, vl = train.iloc[tr_idx], train.iloc[val_idx]
        logdir = exp_dir / f"fold_{fold_num}"
        logdir.mkdir(exist_ok=True)
        trained_model = train_one_fold(tr, vl, hparams, neptune_logger, logdir)
        val_preds = get_predictions(trained_model, vl)
        np.save(str(logdir / "val_preds.npy"), val_preds)
        break