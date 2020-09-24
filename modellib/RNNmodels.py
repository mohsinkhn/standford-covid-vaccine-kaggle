"""RNN based models."""

import math
import torch
from torch import nn

from constants import Mappings


def onehot(sequence, num_tokens):
    bs, seq_len = sequence.size()[:2]
    y_onehot = torch.FloatTensor(bs, seq_len, num_tokens).to(sequence.device)
    y_onehot.zero_()
    y_onehot.scatter_(2, sequence.unsqueeze(2), 1)
    return y_onehot


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Conv1dBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=2, drop=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bnorm1 = nn.BatchNorm1d(out_channels)
        self.gelu = nn.GELU()
        self.drop = nn.Dropout2d(drop)

    def forward(self, x):
        return self.drop(self.gelu(self.bnorm1(self.conv1(x))))


class Conv2dBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1, padding=3, drop=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bnorm1 = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()
        self.drop = nn.Dropout2d(drop)

    def forward(self, x):
        return self.drop(self.gelu(self.bnorm1(self.conv1(x))))


class Conv1dBnStack(nn.Module):
    def __init__(
        self, in_channels, conv_channels, kernel_size=5, stride=1, padding=2, drop=0.2, use_codon=False,
    ):
        super().__init__()
        self.conv_layers = []
        self.use_codon = use_codon
        self.conv_channels = conv_channels
        if use_codon:
            self.codon_conv = nn.Conv1d(
                in_channels=in_channels, out_channels=self.conv_channels[0], stride=3, kernel_size=3, padding=1,
            )
            self.codon_layer = nn.Sequential(
                self.codon_conv, nn.Upsample(scale_factor=2.975, mode="nearest"), nn.Tanh(),
            )
        self.conv0 = Conv1dBn(
            in_channels=in_channels,
            out_channels=self.conv_channels[0],
            stride=stride,
            kernel_size=kernel_size,
            drop=drop,
            padding=padding,
        )
        for i in range(1, len(self.conv_channels)):
            layer = Conv1dBn(
                in_channels=self.conv_channels[i - 1],
                out_channels=self.conv_channels[i],
                stride=stride,
                kernel_size=kernel_size,
                drop=0.2,
                padding=padding,
            )
            self.conv_layers.append(layer)
        self.conv = nn.Sequential(*self.conv_layers)

    def forward(self, x):
        x = x.permute(0, 2, 1).contiguous()
        out = self.conv0(x)
        if self.use_codon:
            out = out + out * self.codon_layer(x)
        out = self.conv(out)
        return out.permute(0, 2, 1).contiguous()


class ParamModel(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.init_params()

    def init_params(self):
        self.num_seq_tokens = len(Mappings.sequence_token2int.keys()) + 1
        self.num_struct_tokens = len(Mappings.structure_token2int.keys()) + 1
        self.num_pl_tokens = len(Mappings.pl_token2int.keys()) + 1
        self.num_pl2_tokens = len(Mappings.pl2_token2int.keys()) + 1
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
        self.conv_channels = self.hparams.get("conv_channels", [256, 512, 256, 512, 256])
        self.bpp_conv_channels = self.hparams.get("bpp_conv_channels", [8, 16, 32, 64])
        self.kernel_size = self.hparams.get("kernel_size", 5)
        self.stride = self.hparams.get("stride", 1)
        self.bpp_thresh = self.hparams.get("bpp_thresh", 0.5)
        self.use_one_hot = self.hparams.get("use_one_hot", False)
        self.add_bpp = self.hparams.get("add_bpp", True)
        self.rnn_type = self.hparams.get("rnn_type", "gru")
        self.conv_drop = self.hparams.get("conv_drop", 0.3)
        self.prob_thresh = self.hparams.get("prob_thresh", 0.5)
        self.use_codon = self.hparams.get("use_codon", False)
        self.pos_dim = self.hparams.get("pos_dim", 128)


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


class RCNNGRUModelv3(ParamModel):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.hparams = hparams
        self.sequence_embedding = nn.Embedding(self.num_seq_tokens, self.seq_emb_dim)
        self.structure_embedding = nn.Embedding(self.num_struct_tokens, self.struct_emb_dim)
        self.predicted_loop_embedding = nn.Embedding(self.num_pl_tokens, self.pl_emb_dim)

        if self.use_one_hot:
            self.seq_conv = Conv1dBnStack(self.num_seq_tokens * 2, self.conv_channels)
            self.struct_pl_conv = Conv1dBnStack(self.num_struct_tokens + self.num_pl_tokens, self.conv_channels)

        else:
            self.seq_conv = Conv1dBnStack(self.seq_emb_dim * 2, self.conv_channels)
            self.struct_pl_conv = Conv1dBnStack(self.struct_emb_dim + self.pl_emb_dim, self.conv_channels)

        self.cont_bnorm = nn.BatchNorm1d(2)
        self.cont_fc = nn.Sequential(nn.Linear(2, 64), nn.ReLU())
        bpp_conv_channels = [1] + self.bpp_conv_channels
        self.bpp_conv = nn.Sequential(*[Conv2dBn(bpp_conv_channels[i], bpp_conv_channels[i+1])
                                       for i in range(len(bpp_conv_channels)-1)])
        if self.rnn_type == "gru":
            rnn_layer = nn.GRU
        else:
            rnn_layer = nn.LSTM
        self.gru = rnn_layer(
            input_size=self.conv_channels[-1] * 2 + self.bpp_conv_channels[-1] * 2,
            hidden_size=self.gru_dim,
            bidirectional=self.bidirectional,
            batch_first=True,
            num_layers=self.gru_layers,
            dropout=self.dropout_prob,
        )
        # self.pos = PositionalEncoding(self.pos_dim)
        self.drop = nn.Dropout2d(self.spatial_dropout)
        self.fc = nn.Sequential(nn.Linear(self.gru_dim * (1 + self.bidirectional), self.target_dim))

    def forward(self, xinputs):
        if self.use_one_hot:
            xseq = onehot(xinputs["sequence"], self.num_seq_tokens)
            xstruct = onehot(xinputs["structure"], self.num_struct_tokens)
            xpl = onehot(xinputs["predicted_loop_type"], self.num_pl_tokens)
            xpairseq = onehot(xinputs["pair_sequence"], self.num_seq_tokens)
        else:
            xseq = self.sequence_embedding(xinputs["sequence"])
            xstruct = self.structure_embedding(xinputs["structure"])
            xpl = self.predicted_loop_embedding(xinputs["predicted_loop_type"])
            xpairseq = self.sequence_embedding(xinputs["pair_sequence"])

        xseq = self.seq_conv(torch.cat([xseq, xpairseq], dim=-1))
        xpl = self.struct_pl_conv(torch.cat([xpl, xstruct], dim=-1))

        bs, seq_len = xseq.size()[:2]

        xbpp = self.bpp_conv(xinputs["bpps"].unsqueeze(1))
        xbpp_mean = nn.AdaptiveAvgPool2d((seq_len, 1))(xbpp).squeeze(3).permute(0, 2, 1).contiguous()
        xbpp_max = nn.AdaptiveMaxPool2d((seq_len, 1))(xbpp).squeeze(3).permute(0, 2, 1).contiguous()
        # xseq = xseq  + xseq * xbpp_mean
        # xpl = xpl + xpl * xbpp_mean
        cont_emb = torch.cat([xbpp_mean, xbpp_max], dim=-1)
        # pos = torch.zeros(bs, seq_len, self.pos_dim, device=xseq.device)
        # pos = self.pos(pos)
        x = torch.cat([xseq, xpl, cont_emb], dim=-1)
        x, _ = self.gru(x)
        x = x[:, : self.max_seq_pred, :]
        x = x.permute(0, 2, 1).contiguous()
        x = self.drop(x)
        x = x.permute(0, 2, 1).contiguous()
        x = self.fc(x)
        return x
