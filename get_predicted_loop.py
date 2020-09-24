import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def predict_loop(sequence, structure):
    !echo $sequence > a.dbn
    !echo "$structure" >> a.dbn
    !perl bpRNA/bpRNA.pl a.dbn
    with open("a.st") as stf:
        result = [l.strip('\n') for l in stf]
    pl = result[5]
    # segments = [row  for row in result if row.startswith('segment')]
    # num_seg, num_bp = len(segments), sum([int(seg.split(" ")[1].strip("bp")) for seg in segments])    
    return pl


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--structure_filepath", required=True)
    parser.add_argument("--pkg", required=True)
    parser.add_argument("--T", required=True)
    T = args.T
    filepath = Path(args.structure_filepath)
    filestem = filepath.stem
    pkg = args.pkg
    vienna_pl = {}

    train = pd.read_json("data/train.json", lines=True)
    test = pd.read_json("data/test.json", lines=True)
    sequences = train.sequence.tolist() + test.sequence.tolist()
    seq_ids = train.id.tolist() + test.id.tolist()

    vienna_structs = pd.read_csv(filepath)

    structures = vienna_structs[f"vienna_{T}"].tolist()
    res = []
    for seq, struc in tqdm(zip(sequences, structures)):
        res.append(predict_loop(seq, struc))
    res = np.array(res)
    df = pd.DataFrame({"{pkg}_pl": res, "id": seq_ids})
    df.to_csv(f"data/{filestem}_{T}_pl.csv")
