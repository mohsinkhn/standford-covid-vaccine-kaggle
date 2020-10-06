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
from sklearn.model_selection import KFold
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torch.nn.functional as F
# from torchcontrib.optim import SWA

from constants import FilePaths, TGT_COLS, Mappings
from datasets import BPPSData
from modellib import RNNmodels


class VAELoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, recon_x, x, mu, logvar):
        if isinstance(recon_x, list):
            BCE = 0
            for recon_x_one in recon_x:
                BCE += F.mse_loss(recon_x_one, x)  # F.binary_cross_entropy_with_logits(recon_x_one, x)
            BCE /= len(recon_x)
        else:
            BCE = F.mse_loss(recon_x, x)  # F.binary_cross_entropy_with_logits(recon_x, x)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD /= BATCH_SIZE * 128
        return BCE + 1e-5 * KLD

def onehot(sequence, num_tokens):
    bs, seq_len = sequence.size()[:2]
    y_onehot = torch.FloatTensor(bs, seq_len, num_tokens).to(sequence.device)
    y_onehot.zero_()
    y_onehot.scatter_(2, sequence.unsqueeze(2), 1)
    return y_onehot


class CustomRunner(dl.Runner):
    def _handle_batch(self, batch):
        x, y = batch
        num_seq_tokens = len(Mappings.sequence_token2int.keys()) + 1
        num_struct_tokens = len(Mappings.structure_token2int.keys()) + 1
        num_pl_tokens = len(Mappings.pl_token2int.keys()) + 1
        xseq = onehot(x["sequence"], num_seq_tokens)
        xstruct = onehot(x["structure"], num_struct_tokens)
        xpl = onehot(x["predicted_loop_type"], num_pl_tokens)
        x = torch.cat([xseq, xstruct, xpl], -1)
        recon_x, mu, logvar = self.model(x)
        loss = self.criterion(recon_x, x, mu, logvar)
        batch_metrics = {"loss": loss}

        if self.is_train_loader:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        # self.input["targets"] = y
        self.output = recon_x
        self.batch_metrics.update(**batch_metrics)


def train_one_fold(tr, vl, hparams, logger, logdir, device):
    tr_ds = BPPSData(
        tr,
        bpps_path="data/bpps",
    )
    vl_ds = BPPSData(vl, bpps_path="data/bpps")
    tr_dl = DataLoader(tr_ds, shuffle=True, drop_last=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,)
    vl_dl = DataLoader(vl_ds, shuffle=False, drop_last=False, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,)

    model = RNNmodels.SequenceEncoder(hparams)
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
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, factor=0.5, min_lr=1e-3)
    if hparams.get("scheduler", "reducelrplateau") == "onecycle":
        total_steps = hparams.get("num_epochs") * (len(tr) // hparams.get("batch_size"))
        max_lr = hparams.get("lr", 1e-3)
        scheduler = OneCycleLRWithWarmup(
            optimizer, num_steps=total_steps, lr_range=(max_lr, max_lr / 10, max_lr / 100), warmup_fraction=0.5,
        )

    criterion = VAELoss()
    runner = CustomRunner(device=device)
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
    )
    return model, tr_dl, vl_dl


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
    train = pd.read_json(FP.train_json, lines=True)[["sequence", "predicted_loop_type", "id", "structure"]]
    test = pd.read_json(FP.train_json, lines=True)[["sequence", "predicted_loop_type", "id", "structure"]]
    data = pd.concat([train, test]).reset_index(drop=True)
    cvlist = list(KFold(20, shuffle=False).split(train))
    m = len(cvlist)
    with open(str(exp_dir / "hparams.json"), "w") as hf:
        json.dump(hparams, hf)

    run_on_single = hparams.get("run_on_single", False)
    device = utils.get_device()
    val_preds = np.zeros(shape=(len(train), hparams["max_seq_pred"], hparams["num_features"]), dtype="float64",)
    for fold_num in [m-1]:
        tr_idx, val_idx = cvlist[fold_num]
        tr, vl = data, data.iloc[val_idx]
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
        if run_on_single:
            break

        if fold_num + 1 != len(cvlist):
            neptune_logger.experiment.stop()
