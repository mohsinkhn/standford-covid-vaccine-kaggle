import argparse
import json

from catalyst.dl import utils
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from constants import FilePaths, TGT_COLS
from datasets import RNAAugDatav2
from modellib import RNNmodels
from nn_trainer import get_predictions


def predict_on_fold(data, fold, device, model, max_seq_pred=107, batch_size=100, num_workers=4):
    ds = RNAAugDatav2(data, targets=None, bpps_path="data/bpps")
    loader = DataLoader(ds, shuffle=False, drop_last=False, batch_size=batch_size, num_workers=num_workers)
    model.max_seq_pred = max_seq_pred
    return get_predictions(model, loader, device)


def predict_on_test(model_folder, public_test, private_test):
    model_path = Path(model_folder)
    folds = model_path.glob("fold_*")
    folds = [int(fold.stem.split("_")[1]) for fold in folds]
    with open(str(model_path / "hparams.json"), "r") as fp:
        hparams = json.load(fp)

    device = utils.get_device()
    public_preds = []
    private_preds = []
    for fold in folds:
        model = getattr(RNNmodels, hparams.get("model_name", "RNAGRUModelv3"))(hparams)
        model.load_state_dict(torch.load(str(model_path / f"fold_{fold}" / "checkpoints/best.pth"))["model_state_dict"])
        pupreds = predict_on_fold(public_test, fold, device, model, 107, BATCH_SIZE, NUM_WORKERS)
        pvpreds = predict_on_fold(private_test, fold, device, model, 130, BATCH_SIZE, NUM_WORKERS)
        public_preds.append(pupreds)
        private_preds.append(pvpreds)
    return np.mean(public_preds, 0), np.mean(private_preds, 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_folder", required=True)
    args = parser.parse_args()
    model_folder = Path(args.model_folder)
    model_tags = model_folder.name

    NUM_WORKERS = 8
    BATCH_SIZE = 32
    FP = FilePaths("data")
    test = pd.read_json(FP.test_json, lines=True)
    public_test = test.loc[test.seq_length == 107]
    private_test = test.loc[test.seq_length == 130]

    public_preds, private_preds = predict_on_test(str(model_folder), public_test, private_test)
    sample_df = pd.read_csv(FP.sample_submission_path)
    preds_ls = []

    for df, preds in [(public_test, public_preds), (private_test, private_preds)]:
        for i, uid in enumerate(df.id):
            single_pred = preds[i]

            single_df = pd.DataFrame(single_pred, columns=TGT_COLS+["deg_pH10", "deg_50C"])
            single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]

            preds_ls.append(single_df)

    preds_df = pd.concat(preds_ls)
    submission = sample_df[['id_seqpos']].merge(preds_df, on=['id_seqpos'])
    submission.to_csv(f"data/submission_{model_tags}.csv", index=False)
