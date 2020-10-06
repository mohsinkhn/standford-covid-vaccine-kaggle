import json

import numpy as np
import pandas as pd
import RNA
from tqdm import tqdm

from constants import FilePaths


def get_entropy(seq):
    fc = RNA.fold_compound(seq)
    mfe_struct, mfe = fc.mfe()
    fc.exp_params_rescale(mfe)
    pp, fp = fc.pf()
    entropy = fc.positional_entropy()
    return mfe, entropy[1:]


if __name__ == "__main__":
    FP = FilePaths("data")
    train = pd.read_json(FP.train_json, lines=True)
    test = pd.read_json(FP.test_json, lines=True)

    seq_ids = train.id.tolist() + test.id.tolist()
    seqs = train.sequence.tolist() + test.sequence.tolist()

    seq_entropy = {}
    seq_mfe = {}
    for seqid, seq in tqdm(zip(seq_ids, seqs)):
        mfe, ent = get_entropy(seq)
        seq_entropy[seqid] = ent
        seq_mfe[seqid] = mfe

    with open("data/sequence_entropy.json", "w") as fp:
        json.dump(seq_entropy, fp)

    with open("data/sequence_mfe.json", "w") as fp:
        json.dump(seq_mfe, fp)

    print(np.mean(list(seq_mfe.values())))
    print(np.mean(list(seq_entropy.values())))
