"""RNN based models."""

from torch import nn

from constants import Token2Int


class RNAGRUModel(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.init_params()
        self.embedding = nn.Embedding(len(Token2Int.keys()), self.emb_dim)
        self.gru = nn.GRU(
            input_size=self.emb_dim * self.num_features,
            hidden_size=self.gru_dim,
            bidirectional=self.bidirectional,
            batch_first=True,
            num_layers=self.gru_layers,
            dropout=self.dropout_prob
        )
        self.drop = nn.Dropout2d(self.dropout_prob)
        self.fc = nn.Sequential(nn.Linear(self.gru_dim * (1 + self.bidirectional), self.target_dim))

    def init_params(self):
        self.emb_dim = self.hparams.get("emb_dim", 64)
        self.gru_dim = self.hparams.get("gru_dim", 32)
        self.bidirectional = self.hparams.get("bidirectional", True)
        self.target_dim = self.hparams.get("target_dim", 5)
        self.num_features = self.hparams.get("num_features", 3)
        self.max_seq_pred = self.hparams.get("max_seq_pred", 68)
        self.dropout_prob = self.hparams.get("dropout_prob", 0.2)
        self.gru_layers = self.hparams.get("gru_layers", 3)

    def forward(self, x):
        x = self.embedding(x)
        bs, seq_len = x.size()[:2]
        x = x.contiguous().view(bs, seq_len, -1)
        x, _ = self.gru(x)
        x = x[:, : self.max_seq_pred, :]
        x = x.permute(0, 2, 1)
        x = self.drop(x)
        x = x.permute(0, 2, 1)
        x = self.fc(x)
        return x
