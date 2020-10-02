import argparse
import json

from catalyst.dl import utils
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch_geometric.data import DataLoader

from constants import FilePaths, TGT_COLS
from pytorch_geometric_dataset import GraphDataset
from modellib import graphmodel
from gnn_trainer_v2 import get_predictions_gcn


def predict_on_fold(data, fold, device, model, hparams, batch_size=100, num_workers=4):
    ds = GraphDataset(data, False, hparams)
    loader = DataLoader(ds.data, shuffle=False, batch_size=batch_size, num_workers=num_workers)
    # model.max_seq_pred = max_seq_pred
    return get_predictions_gcn(model, loader, device)


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
        print(fold)
        hparams["num_layers"] = 5
        if fold == 0:
            continue
            hparams["num_layers"] = 10
        model = getattr(graphmodel, hparams.get("model_name", "HyperNodeGCN"))(hparams)
        model.load_state_dict(torch.load(str(model_path / f"fold_{fold}" / "checkpoints/best.pth"))["model_state_dict"])
        pupreds = predict_on_fold(public_test, fold, device, model, hparams, BATCH_SIZE, NUM_WORKERS)
        pvpreds = predict_on_fold(private_test, fold, device, model, hparams, BATCH_SIZE, NUM_WORKERS)
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

            single_df = pd.DataFrame(single_pred, columns=TGT_COLS)
            single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]
            single_df["deg_pH10"] = 0
            single_df["deg_50C"] = 0
            preds_ls.append(single_df)

    preds_df = pd.concat(preds_ls)
    submission = sample_df[['id_seqpos']].merge(preds_df, on=['id_seqpos'])
    submission.to_csv(f"data/submission_{model_tags}.csv", index=False)
