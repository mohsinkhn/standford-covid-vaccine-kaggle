import json
import pandas as pd
from pathlib import Path
import numpy as np
from tqdm import tqdm

from arnie.mea.mea import MEA
from constants import FilePaths


def predict_MEA_structures(matrix_list, gamma_min=-7, gamma_max=7):
    matrices = [np.load(x) for x in matrix_list]
    seq_id = [Path(x).stem for x in matrix_list]
    gamma_vals = [x for x in range(gamma_min, gamma_max)]
    structures = {}
    struct_scores = {}
    for sid, matrix in tqdm(zip(seq_id, matrices)):
        row_metrics = []
        row_structure = []
        for g in gamma_vals:
            mea_cls = MEA(matrix, gamma=2**g)
            metrics = mea_cls.score_expected() #sen, ppv, mcc, fscore
            row_metrics.append(metrics)
            row_structure.append(mea_cls.structure)
        struct_scores[sid] = row_metrics
        structures[sid] = row_structure
    return structures, struct_scores


if __name__ == "__main__":
    FP = FilePaths("data")
    train = pd.read_json(FP.train_json, lines=True)
    test = pd.read_json(FP.test_json, lines=True)

    ids = train.id.tolist() + test.id.tolist()
    matrix_list = [f"data/bpps/{sid}.npy" for sid in ids]
    structures, scores = predict_MEA_structures(matrix_list)

    with open("data/additional_structures.json", "w") as f:
        json.dump(structures, f)
    
    with open("data/additional_structure_scores.json", "w") as f:
        json.dump(scores, f)
