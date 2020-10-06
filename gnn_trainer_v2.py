import argparse
import json
import os
import random

from catalyst import dl
from catalyst.dl import utils
from catalyst.core.callbacks.scheduler import SchedulerCallback
from catalyst.contrib.dl.callbacks.neptune_logger import NeptuneLogger
from catalyst.contrib.nn.schedulers.onecycle import OneCycleLRWithWarmup
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
import torch
from torch_geometric.data import DataLoader
from tqdm import tqdm

from constants import FilePaths, TGT_COLS
from nn_trainer import calc_error_mean, MCMSRE, MCRMSE, mcrmse
from pytorch_geometric_dataset import GraphDataset
from modellib import graphmodel


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


class CustomRunner(dl.Runner):
    # def predict_batch(self, batch):
    # model inference step
    #    return self.model(batch[0].to(self.device).view(batch[0].size(0), -1))
    def _run_batch(self, batch) -> None:
        """
        Inner method to run train step on specified data batch,
        with batch callbacks events.
        Args:
            batch (Mapping[str, Any]): dictionary with data batches
                from DataLoader.
        """
        if isinstance(batch, dict):
            self.batch_size = len(next(iter(batch.values())))
        else:
            self.batch_size = int(batch.batch[-1]) + 1
        self.global_sample_step += self.batch_size
        self.loader_sample_step += self.batch_size
        # batch = self._batch2device(batch, self.device)
        batch = batch.to(self.device)
        self.input = batch

        self._run_event("on_batch_start")
        self._handle_batch(data=batch)
        self._run_event("on_batch_end")

    def _handle_batch(self, data):
        # model train/valid step
        mask = data.train_mask
        y_hat = self.model(data)[mask]
        y = data.y[mask]
        y_hat = y_hat.reshape(-1, 68, 3)
        y = y.reshape(-1, 68, 3)
        loss = self.criterion(y_hat, y)
        batch_metrics = {"loss": loss.item()}

        if self.is_train_loader:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        # self.input["targets"] = y
        self.output = y_hat
        self.batch_metrics.update(**batch_metrics)



def train_one_fold(tr, vl, hparams, logger, logdir, device):
    tr_ds = GraphDataset(tr, True, hparams, "data/predicted_loop_segments.csv", ["data/bpps"])
    vl_ds = GraphDataset(vl, True, hparams, "data/predicted_loop_segments.csv", ["data/bpps"])

    tr_dl = DataLoader(tr_ds.data, shuffle=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    vl_dl = DataLoader(vl_ds.data, shuffle=False, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    model = getattr(graphmodel, hparams.get("model_name", "HyperNodeGCN"))(hparams)
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
        callbacks=[logger, SchedulerCallback(mode="epoch"),
                   # dl.OptimizerCallback(metric_key="loss")
                ],
        load_best_on_end=True,
        # resume="logs/gcn__transformer__hypernode__run3/fold_4/checkpoints/best_full.pth"
    )
    return model, tr_dl, vl_dl


def get_predictions_gcn(model, loader, device):
    model.eval()
    model.to(device)
    preds = []
    with torch.no_grad():
        for data in tqdm(loader):
            data.to(device)
            bs = int(data.batch[-1]) + 1
            b_preds = model(data)[data.test_mask]
            b_preds = b_preds.cpu()
            b_preds = b_preds.numpy()
            b_preds = b_preds.reshape(bs, -1, 3)
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

    set_seed(hparams["seed"])
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
        vds = GraphDataset(train.iloc[val_idx], True, hparams, "data/predicted_loop_segments.csv", ["data/bpps"])
        vdl = DataLoader(vds.data, shuffle=False, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,)
        val_pred = get_predictions_gcn(trained_model, vdl, device)[:, :68, : hparams["num_features"]]
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
