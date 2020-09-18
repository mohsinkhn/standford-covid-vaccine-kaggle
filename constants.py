from pathlib import Path


ROOT = Path("data")
BPPS_PATH = ROOT / "bpss"
TRAIN_JSON = ROOT / "train.json"
TEST_JSON = ROOT / "test.json"
SAMPLE_SUB = ROOT / "sample_submission.csv"


Token2Int = {s: i for i, s in enumerate("(.)ACGUBEHIMSX")}


class Mappings(object):
    structure_token2int = {s: i for i, s in enumerate("(.)")}
    sequence_token2int = {s: i for i, s in enumerate("ACGU")}
    pl_token2int = {s: i for i, s in enumerate("BEHIMSX")}
