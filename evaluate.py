import argparse
import json

from catalyst.dl import utils
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
import torch
from torch.utils.data import DataLoader

# from torchcontrib.optim import SWA

from constants import FilePaths, TGT_COLS
from datasets import RNAAugDatav2
from modellib import RNNmodels
from nn_trainer import get_predictions, validation_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_folder", required=True)
    args = parser.parse_args()

    model_path = Path(args.model_folder)
    folds = model_path.glob("fold_*")
    folds = [int(fold.stem.split("_")[1]) for fold in folds]
    with open(str(model_path / "hparams.json"), "r") as fp:
        hparams = json.load(fp)

    NUM_WORKERS = 8
    BATCH_SIZE = 4  # hparams.get("batch_size", 32)
    FP = FilePaths("data")
    train = pd.read_json(FP.train_json, lines=True)
    cvlist = list(
        StratifiedKFold(hparams.get("num_folds"), shuffle=True, random_state=hparams.get("seed")).split(
            train, train["SN_filter"]
        )
    )

    device = "cpu"  # utils.get_device()
    val_preds = np.zeros(shape=(len(train), hparams["max_seq_pred"], hparams["num_features"]), dtype="float64")
    for fold in folds:
        val_idx = cvlist[fold][1]
        vl = train.iloc[val_idx]
        vl_ds = RNAAugDatav2(vl, targets=TGT_COLS)
        vl_dl = DataLoader(vl_ds, shuffle=False, drop_last=False, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
        model = getattr(RNNmodels, hparams.get("model_name", "RNAGRUModelv3"))(hparams)
        model.load_state_dict(torch.load(model_path / f"fold_{fold}" / "checkpoints/best.pth", map_location="cpu")["model_state_dict"])
        val_preds[val_idx] = get_predictions(model, vl_dl, device)[:, :, : hparams["num_features"]]

    np.save(str(model_path / "val_preds.npy"), val_preds)
    y_trues = np.dstack((np.vstack(train[col].values) for col in TGT_COLS))
    sn_flag = train["SN_filter"].values.astype(bool)
    eval_results = validation_metrics(y_trues, val_preds, sn_flag)
    print(eval_results)
