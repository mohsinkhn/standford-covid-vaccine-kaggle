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
        self.num_seq_tokens = len(Mappings.sequence_token2int.keys()) + 1
        self.num_struct_tokens = len(Mappings.structure_token2int.keys()) + 1
        self.num_pl_tokens = len(Mappings.pl_token2int.keys()) + 1
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
        self.bpp_thresh = self.hparams.get("bpp_thresh", 0.5)


class RNAGRUModel(ParamModel):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.hparams = hparams
        self.sequence_embedding = nn.Embedding(self.num_seq_tokens, self.seq_emb_dim)
        self.structure_embedding = nn.Embedding(self.num_struct_tokens, self.struct_emb_dim)
        self.predicted_loop_embedding = nn.Embedding(self.num_pl_tokens, self.pl_emb_dim)
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
        xpl = self.predicted_loop_embedding(x["predicted_loop_type"])
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
        self.predicted_loop_embedding = nn.Embedding(self.num_pl_tokens, self.pl_emb_dim)
        self.conv = Conv1dBn(in_channels=(self.seq_emb_dim + self.struct_emb_dim + self.pl_emb_dim)*2 + 1,
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

    def forward(self, xinputs):
        xseq = self.sequence_embedding(xinputs["sequence"])
        xstruct = self.structure_embedding(xinputs["structure"])
        xpl = self.predicted_loop_embedding(xinputs["predicted_loop_type"])

        x = torch.cat((xseq, xstruct, xpl), dim=-1).squeeze(2)
        xbpp_prob, xbpp_idx = xinputs["bpps"].max(dim=1)
        xbpp_idx = xbpp_idx.unsqueeze(2)
        xbpp_prob = xbpp_prob.unsqueeze(2)

        mask = (xbpp_prob < self.bpp_thresh)

        p_xseq = xinputs["sequence"].gather(1, xbpp_idx)
        p_xseq_ = p_xseq.masked_fill(mask, 0)
        p_xseq_emb = self.sequence_embedding(p_xseq_).squeeze(2)

        p_xstruct = xinputs["structure"].gather(1, xbpp_idx)
        p_xstruct_ = p_xstruct.masked_fill(mask, 0)
        p_xstruct_emb = self.structure_embedding(p_xstruct_).squeeze(2)

        p_xpl = xinputs["predicted_loop_type"].gather(1, xbpp_idx)
        p_xpl_ = p_xpl.masked_fill(mask, 0)
        p_xpl_emb = self.predicted_loop_embedding(p_xpl_).squeeze(2)
        x2 = torch.cat((p_xseq_emb, p_xstruct_emb, p_xpl_emb, xbpp_prob), dim=-1)
        x = torch.cat((x, x2), dim=-1)
        x = x.permute(0, 2, 1).contiguous()
        x = self.conv(x)
        x = x.permute(0, 2, 1).contiguous()
        x, _ = self.gru(x)
        x = x[:, : self.max_seq_pred, :]
        x = x.permute(0, 2, 1).contiguous()
        x = self.drop(x)
        x = x.permute(0, 2, 1).contiguous()
        x = self.fc(x)
        return x
