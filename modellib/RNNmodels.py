"""RNN based models."""

import torch
from torch import nn

from constants import Mappings


class Conv1dBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bnorm1 = nn.BatchNorm1d(out_channels)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.gelu(self.bnorm1(self.conv1(x)))


class ParamModel(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.init_params()

    def init_params(self):
        self.num_seq_tokens = len(Mappings.sequence_token2int.keys())
        self.num_struct_tokens = len(Mappings.structure_token2int.keys())
        self.num_pl_tokens = len(Mappings.pl_token2int.keys())
        self.seq_emb_dim = self.hparams.get("seq_emb_dim", 32)
        self.struct_emb_dim = self.hparams.get("struct_emb_dim", 32)
        self.pl_emb_dim = self.hparams.get("pl_emb_dim", 32)
        self.gru_dim = self.hparams.get("gru_dim", 32)
        self.bidirectional = self.hparams.get("bidirectional", True)
        self.target_dim = self.hparams.get("target_dim", 5)
        self.num_features = self.hparams.get("num_features", 3)
        self.max_seq_pred = self.hparams.get("max_seq_pred", 68)
        self.dropout_prob = self.hparams.get("dropout_prob", 0.2)
        self.spatial_dropout = self.hparams.get("spatial_dropout", 0.2)
        self.gru_layers = self.hparams.get("gru_layers", 3)
        self.conv_channels = self.hparams.get("conv_channels", 256)
        self.kernel_size = self.hparams.get("kernel_size", 5)
        self.stride = self.hparams.get("stride", 1)


class RNAGRUModel(ParamModel):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.hparams = hparams
        self.sequence_embedding = nn.Embedding(self.num_seq_tokens, self.seq_emb_dim)
        self.structure_embedding = nn.Embedding(self.num_struct_tokens, self.struct_emb_dim)
        self.predictd_loop_embedding = nn.Embedding(self.num_pl_tokens, self.pl_emb_dim)
        self.gru = nn.GRU(
            input_size=(self.seq_emb_dim + self.struct_emb_dim + self.pl_emb_dim),
            hidden_size=self.gru_dim,
            bidirectional=self.bidirectional,
            batch_first=True,
            num_layers=self.gru_layers,
            dropout=self.dropout_prob,
        )
        self.drop = nn.Dropout2d(self.spatial_dropout)
        self.fc = nn.Sequential(nn.Linear(self.gru_dim * (1 + self.bidirectional), self.target_dim))

    def forward(self, x):
        xseq = self.sequence_embedding(x["sequence"])
        xstruct = self.structure_embedding(x["structure"])
        xpl = self.predictd_loop_embedding(x["predicted_loop_type"])
        x = torch.cat((xseq, xstruct, xpl), dim=-1).squeeze(2)
        x, _ = self.gru(x)
        x = x[:, : self.max_seq_pred, :]
        x = x.contiguous().permute(0, 2, 1)
        x = self.drop(x)
        x = x.contiguous().permute(0, 2, 1)
        x = self.fc(x)
        return x


class RCNNGRUModel(ParamModel):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.hparams = hparams
        self.sequence_embedding = nn.Embedding(self.num_seq_tokens, self.seq_emb_dim)
        self.structure_embedding = nn.Embedding(self.num_struct_tokens, self.struct_emb_dim)
        self.predictd_loop_embedding = nn.Embedding(self.num_pl_tokens, self.pl_emb_dim)
        self.conv = Conv1dBn(in_channels=self.seq_emb_dim + self.struct_emb_dim + self.pl_emb_dim,
                             out_channels=self.conv_channels, stride=self.stride, kernel_size=self.kernel_size)
        self.gru = nn.GRU(
            input_size=self.conv_channels,
            hidden_size=self.gru_dim,
            bidirectional=self.bidirectional,
            batch_first=True,
            num_layers=self.gru_layers,
            dropout=self.dropout_prob,
        )
        self.drop = nn.Dropout2d(self.spatial_dropout)
        self.fc = nn.Sequential(nn.Linear(self.gru_dim * (1 + self.bidirectional), self.target_dim))

    def forward(self, x):
        xseq = self.sequence_embedding(x["sequence"])
        xstruct = self.structure_embedding(x["structure"])
        xpl = self.predictd_loop_embedding(x["predicted_loop_type"])
        x = torch.cat((xseq, xstruct, xpl), dim=-1).squeeze(2)
        x = x.contiguous().permute(0, 2, 1)
        x = self.conv(x)
        x = x.contiguous().permute(0, 2, 1)
        x, _ = self.gru(x)
        x = x[:, : self.max_seq_pred, :]
        x = x.contiguous().permute(0, 2, 1)
        x = self.drop(x)
        x = x.contiguous().permute(0, 2, 1)
        x = self.fc(x)
        return x
