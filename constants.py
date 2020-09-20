from pathlib import Path


class FilePaths:
    def __init__(self, input_path):
        self.input_path = Path(input_path)
        self.bpps_path = self.input_path / "bpps"
        self.train_json = self.input_path / "train.json"
        self.test_json = self.input_path / "test.json"
        self.sample_submission_path = self.input_path / "sample_submission.csv"


class Mappings:
    structure_token2int = {s: i + 1 for i, s in enumerate("(.)")}
    sequence_token2int = {s: i + 1 for i, s in enumerate("ACGU")}
    pl_token2int = {s: i + 1 for i, s in enumerate("BEHIMSX")}


TGT_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
