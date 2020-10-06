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
    pl2_token2int = {
        s: i + 1
        for i, s in enumerate(
            [
                "0",
                "B1",
                "B10",
                "B11",
                "B12",
                "B13",
                "B2",
                "B3",
                "B4",
                "B5",
                "B6",
                "B7",
                "B8",
                "B9",
                "E1",
                "E2",
                "H1",
                "H2",
                "H3",
                "H4",
                "H5",
                "H6",
                "H7",
                "I1",
                "I11",
                "I13",
                "I14",
                "I15",
                "I17",
                "I19",
                "I2",
                "I20",
                "I21",
                "I23",
                "I25",
                "I26",
                "I27",
                "I29",
                "I3",
                "I32",
                "I35",
                "I38",
                "I41",
                "I5",
                "I7",
                "I8",
                "I9",
                "M1",
                "M11",
                "M2",
                "M3",
                "M4",
                "M5",
                "M6",
                "M7",
                "M8",
                "S1",
                "S10",
                "S11",
                "S12",
                "S13",
                "S14",
                "S15",
                "S16",
                "S2",
                "S3",
                "S4",
                "S5",
                "S6",
                "S7",
                "S8",
                "S9",
                "X1",
                "X2",
                "X3",
                "X4",
                "X5",
            ]
        )
    }


TGT_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
