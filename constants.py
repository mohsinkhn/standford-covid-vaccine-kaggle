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
    pl2_token2int = {s: i + 1 for i, s in enumerate(['f0',
 'h0',
 'h1',
 'h2',
 'h3',
 'h4',
 'h5',
 'h6',
 'i0',
 'i1',
 'i2',
 'i3',
 'i4',
 'i5',
 'i6',
 'i7',
 'i8',
 'i9',
 'm0',
 'm1',
 'm2',
 'm3',
 'm4',
 'm5',
 'm6',
 'm7',
 'm8',
 's0',
 's1',
 's10',
 's11',
 's2',
 's3',
 's4',
 's5',
 's6',
 's7',
 's8',
 's9',
 't0'])}


TGT_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
