"""RNN based models."""
from copy import deepcopy
import json
import math
from gensim.models import KeyedVectors
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

from constants import Mappings
from modellib.transformermodels import CustomTransformerEncoderLayer, PositionalEncoding
from modellib.resnet import ResNet, Bottleneck


def onehot(sequence, num_tokens):
    bs, seq_len = sequence.size()[:2]
    y_onehot = torch.FloatTensor(bs, seq_len, num_tokens).to(sequence.device)
    y_onehot.zero_()
    y_onehot.scatter_(2, sequence.unsqueeze(2), 1)
    return y_onehot


class SpatialDropout(nn.Module):
    def __init__(self, dropout=0.2):
        super().__init__()
        self.drop = nn.Dropout2d(p=dropout)

    def forward(self, x):
        x = x.permute(0, 2, 1).contiguous()
        x = self.drop(x)
        x = x.permute(0, 2, 1).contiguous()
        return x


class Conv1dBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=2, drop=0.1, groups=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups)
        self.bnorm1 = nn.BatchNorm1d(out_channels)
        self.gelu = nn.GELU()
        self.drop = nn.Dropout2d(drop)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bnorm1(x)  # .permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        x = self.drop(self.gelu(x))
        return x


class Conv2dBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=2, drop=0.1):
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
        groups=1
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
        if not isinstance(groups, list):
            groups = len(conv_channels) * [groups]

        self.conv0 = Conv1dBn(
            in_channels=in_channels,
            out_channels=self.conv_channels[0],
            stride=stride,
            kernel_size=kernel_size,
            drop=drop,
            padding=padding,
            groups=groups[0]
        )
        for i in range(1, len(self.conv_channels)):
            layer = Conv1dBn(
                in_channels=self.conv_channels[i - 1],
                out_channels=self.conv_channels[i],
                stride=stride,
                kernel_size=kernel_size,
                drop=drop,
                padding=padding,
                groups=groups[i]
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
        self.intermediate_dim = self.hparams.get("intermediate_dim", 256)
        self.combined_emb_dim = self.hparams.get("combined_emb_dim", 150)
        self.add_segment_info = self.hparams.get("add_segment_info", False)
        self.add_entropy = self.hparams.get("add_entropy", False)
        self.use_6n = self.hparams.get("use_6n", False)
        self.adaptive_lr = self.hparams.get("adaptive_lr", False)


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
        bpp_conv_channels = [15] + self.bpp_conv_channels
        self.bpp_conv = nn.Sequential(
            *[Conv2dBn(bpp_conv_channels[i], bpp_conv_channels[i + 1]) for i in range(len(bpp_conv_channels) - 1)]
        )
        if self.rnn_type == "gru":
            rnn_layer = nn.GRU
        else:
            rnn_layer = nn.LSTM
        self.gru = rnn_layer(
            input_size=self.conv_channels[-1] * 3 + self.bpp_conv_channels[-1] * 2,
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

        xbpp = self.bpp_conv(xinputs["bpps"].permute(0, 3, 1, 2).contiguous())
        xbpp_mean = nn.AdaptiveAvgPool2d((seq_len, 1))(xbpp).squeeze(3).permute(0, 2, 1).contiguous()
        xbpp_max = nn.AdaptiveMaxPool2d((seq_len, 1))(xbpp).squeeze(3).permute(0, 2, 1).contiguous()
        xseqpl = xseq * xpl * xbpp_mean
        # xpl = xpl + xpl * xbpp_mean
        cont_emb = torch.cat([xbpp_mean, xbpp_max], dim=-1)
        # pos = torch.zeros(bs, seq_len, self.pos_dim, device=xseq.device)
        # pos = self.pos(pos)
        x = torch.cat([xseq, xpl, xseqpl, cont_emb], dim=-1)
        x, _ = self.gru(x)
        # x = torch.cat([x, cont_emb], dim=-1)
        x = x[:, : self.max_seq_pred, :]
        x = x.permute(0, 2, 1).contiguous()
        x = self.drop(x)
        x = x.permute(0, 2, 1).contiguous()
        x = self.fc(x)
        return x


class RCNNGRUModelv4(ParamModel):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.hparams = hparams
        self.sequence_embedding = nn.Embedding(self.num_seq_tokens, self.seq_emb_dim)
        self.structure_embedding = nn.Embedding(self.num_struct_tokens, self.struct_emb_dim)
        self.predicted_loop_embedding = nn.Embedding(self.num_pl_tokens, self.pl_emb_dim)
        # self.combined_embedding = nn.Embedding(self.num_seq_tokens+self.num_struct_tokens+self.num_pl_tokens, self.combined_emb_dim)
        if self.use_one_hot:
            self.seq_conv = Conv1dBnStack(self.num_seq_tokens * 2, self.conv_channels)
            self.struct_pl_conv = Conv1dBnStack(self.num_struct_tokens + self.num_pl_tokens, self.conv_channels)

        else:
            self.seq_conv = Conv1dBnStack(self.seq_emb_dim * 2, self.conv_channels)
            self.struct_pl_conv = Conv1dBnStack(self.struct_emb_dim + self.pl_emb_dim, self.conv_channels)

        self.cont_bnorm = nn.BatchNorm1d(2)
        self.cont_fc = nn.Sequential(nn.Linear(2, 64), nn.ReLU())
        bpp_conv_channels = [15] + self.bpp_conv_channels
        self.bpp_bnorm = nn.BatchNorm2d(15)
        self.bpp_conv = nn.Sequential(
            *[Conv2dBn(bpp_conv_channels[i], bpp_conv_channels[i + 1]) for i in range(len(bpp_conv_channels) - 1)]
        )
        # self.bpp_attn = nn.Sequential(
        #     nn.Conv2d(bpp_conv_channels[-1], 1, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(1),
        #     nn.Tanh()
        # )
        # self.struc_attn = nn.Sequential(
        #     nn.Conv1d(self.conv_channels[-1], 1, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm1d(1),
        #     nn.Tanh()
        # )

        if self.rnn_type == "gru":
            rnn_layer = nn.GRU
        else:
            rnn_layer = nn.LSTM
        self.gru = rnn_layer(
            input_size=self.conv_channels[-1] * 3 + self.bpp_conv_channels[-1] * 2 + 1,
            hidden_size=self.gru_dim,
            bidirectional=self.bidirectional,
            batch_first=True,
            num_layers=self.gru_layers,
            dropout=self.dropout_prob,
        )

        # self.pos = PositionalEncoding(self.pos_dim)
        self.drop = SpatialDropout(self.spatial_dropout)
        self.drop2 = SpatialDropout(0.1)
        # self.fc1 = nn.Sequential(, self.intermediate_dim))
        self.fc = nn.Linear(self.gru_dim * (1 + self.bidirectional), self.target_dim)
        # nn.init.kaiming_uniform_(self.fc.weight)

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
        xbpp_inp = xinputs["bpps"].permute(0, 3, 1, 2).contiguous()
        xbpp = self.bpp_conv(xbpp_inp)
        xbpp_mean = nn.AdaptiveAvgPool2d((seq_len, 1))(xbpp).squeeze(3).permute(0, 2, 1).contiguous()
        xbpp_max = nn.AdaptiveMaxPool2d((seq_len, 1))(xbpp).squeeze(3).permute(0, 2, 1).contiguous()
        # xseqpl = xseq * nn.Tanh()(xpl)
        xseqbpp = torch.matmul(xbpp, xseq.unsqueeze(1)).mean(1)
        # xpl = xpl + xpl * xbpp_mean
        cont_emb = torch.cat([xbpp_mean, xbpp_max, xbpp_inp.mean(dim=(1, 2)).unsqueeze(2)], dim=-1)
        # pos = torch.zeros(bs, seq_len, self.pos_dim, device=xseq.device)
        # pos = self.pos(pos)
        x = torch.cat([xseq, xpl, cont_emb, xseqbpp], dim=-1)
        x, _ = self.gru(x)
        # x = torch.cat([x, xbpp_inp.mean(dim=(1, 3)).unsqueeze(2)], dim=-1)
        x = x[:, : self.max_seq_pred, :]
        x = self.drop(x)
        # x = nn.GELU()(self.fc1(x))
        # x = self.drop2(x)
        x = self.fc(x)
        return x


class RCNNGRUModelv5(ParamModel):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.hparams = hparams
        self.sequence_embedding = nn.Embedding(self.num_seq_tokens, self.seq_emb_dim)
        self.structure_embedding = nn.Embedding(self.num_struct_tokens, self.struct_emb_dim)
        self.predicted_loop_embedding = nn.Embedding(self.num_pl_tokens, self.pl_emb_dim)
        if self.use_one_hot:
            self.seq_conv = Conv1dBnStack(self.num_seq_tokens * 2, self.conv_channels)
        else:
            self.seq_conv = Conv1dBnStack(
                self.seq_emb_dim * 2 + self.struct_emb_dim + self.pl_emb_dim, self.conv_channels
            )

        self.cont_bnorm = nn.BatchNorm1d(2)
        self.cont_fc = nn.Sequential(nn.Linear(2, 64), nn.ReLU())
        bpp_conv_channels = [15] + self.bpp_conv_channels
        self.bpp_bnorm = nn.BatchNorm2d(15)
        self.bpp_conv = nn.Sequential(
            *[Conv2dBn(bpp_conv_channels[i], bpp_conv_channels[i + 1]) for i in range(len(bpp_conv_channels) - 1)]
        )

        if self.rnn_type == "gru":
            rnn_layer = nn.GRU
        else:
            rnn_layer = nn.LSTM
        self.gru = rnn_layer(
            input_size=self.conv_channels[-1] * 2 + self.bpp_conv_channels[-1] * 2 + 1,
            hidden_size=self.gru_dim,
            bidirectional=self.bidirectional,
            batch_first=True,
            num_layers=self.gru_layers,
            dropout=self.dropout_prob,
        )
        self.drop = SpatialDropout(self.spatial_dropout)
        self.fc = nn.Linear(self.gru_dim * (1 + self.bidirectional), self.target_dim)

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

        xseq = self.seq_conv(torch.cat([xseq, xpairseq, xpl, xstruct], dim=-1))
        bs, seq_len = xseq.size()[:2]
        xbpp_inp = xinputs["bpps"].permute(0, 3, 1, 2).contiguous()
        xbpp = self.bpp_conv(xbpp_inp)
        xbpp_mean = nn.AdaptiveAvgPool2d((seq_len, 1))(xbpp).squeeze(3).permute(0, 2, 1).contiguous()
        xbpp_max = nn.AdaptiveMaxPool2d((seq_len, 1))(xbpp).squeeze(3).permute(0, 2, 1).contiguous()
        # xbpp_pair = xbpp_mean.gather(xinputs["pair_sequence"], 1)
        xseqbpp = torch.matmul(xbpp, xseq.unsqueeze(1)).mean(1)
        cont_emb = torch.cat([xbpp_mean, xbpp_max, xbpp_inp.mean(dim=(1, 2)).unsqueeze(2)], dim=-1)
        x = torch.cat([xseq, xseqbpp, cont_emb], dim=-1)
        x, _ = self.gru(x)
        x = x[:, : self.max_seq_pred, :]
        x = self.drop(x)
        x = self.fc(x)
        return x


class RCNNGRUModelv6(ParamModel):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.hparams = hparams
        self.sequence_embedding = nn.Embedding(self.num_seq_tokens, self.seq_emb_dim)
        # w2v_model = KeyedVectors.load("data/w2v_seq_6gram.vectors")
        # embeddings = [np.zeros(shape=(32,))] + [w2v_model.wv[str(i)] for i in range(len(w2v_model.vocab))]
        # embeddings = torch.from_numpy(np.vstack(embeddings).astype('float32'))
        # self.sequence_embedding.weight = nn.Parameter(embeddings/8)

        self.structure_embedding = nn.Embedding(self.num_struct_tokens, self.struct_emb_dim)
        if self.use_6n:
            self.sequence_embedding = nn.Embedding(1028, 64)
            embeddings = np.load("data/seq6n_vectors.npy").astype('float32')
            self.sequence_embedding.weight = nn.Parameter(torch.from_numpy(embeddings))

        self.predicted_loop_embedding = nn.Embedding(self.num_pl_tokens, self.pl_emb_dim)
        if self.use_one_hot:
            self.seq_conv = Conv1dBnStack(self.num_seq_tokens * 2, self.conv_channels)
            self.struct_pl_conv = Conv1dBnStack(self.num_struct_tokens + self.num_pl_tokens, self.conv_channels)
        else:
            seq_inp = self.seq_emb_dim * 2
            if self.add_segment_info:
                seq_inp += 86
            self.seq_conv = Conv1dBnStack(seq_inp, self.conv_channels, groups=1, kernel_size=5, padding=2)
            self.struct_pl_conv = Conv1dBnStack(self.struct_emb_dim + self.pl_emb_dim, self.conv_channels, groups=1, kernel_size=5, padding=2)
        seg_conv_channels = [128, 128]
        ent_conv_channels = [32, 128]
        # self.seg_conv = Conv1dBnStack(86, seg_conv_channels, groups=1)
        self.ent_conv = Conv1dBnStack(1, ent_conv_channels, groups=1)

        bpp_conv_channels = [15] + self.bpp_conv_channels
        # self.cont_fc = nn.Sequential(nn.BatchNorm1d(bpp_conv_channels[-1]*2), nn.Linear(bpp_conv_channels[-1]*2, 64), nn.GELU())
        self.bpp_bnorm = nn.BatchNorm2d(15)
        self.bpp_conv = nn.Sequential(
            *[Conv2dBn(bpp_conv_channels[i], bpp_conv_channels[i + 1], kernel_size=1, padding=0) for i in range(len(bpp_conv_channels) - 1)]
        )
        # self.embed_norm = nn.LayerNorm(300)
        if self.rnn_type == "gru":
            rnn_layer = nn.GRU
        else:
            rnn_layer = nn.LSTM
        inp_shape = self.conv_channels[-1] * 3 + 1 # + bpp_conv_channels[-1] * 2
        # if self.add_segment_info:
        #     inp_shape += seg_conv_channels[-1]
        if self.add_entropy:
            inp_shape += ent_conv_channels[-1]
        self.pos = PositionalEncoding(self.gru_dim*(1+self.bidirectional))
        # self.transformer = TransformerCustomEncoder(1, inp_shape, 512, bpp_conv_channels[0], [1], 0.1, 0.1)
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(self.gru_dim*(1+self.bidirectional), 4, 1024, 0.1, "gelu"), 1)

        self.gru = rnn_layer(
            input_size=inp_shape,
            hidden_size=self.gru_dim,
            bidirectional=self.bidirectional,
            batch_first=True,
            num_layers=self.gru_layers,
            dropout=self.dropout_prob,
        )
        self.drop = nn.Dropout(self.spatial_dropout)
        self.drop2 = nn.Dropout(0.5)
        self.fc1 = nn.Sequential(nn.Linear(self.gru_dim * (1 + self.bidirectional), self.gru_dim * (1 + self.bidirectional)), nn.ReLU())
        self.fc = nn.Linear(self.gru_dim * (1 + self.bidirectional), self.target_dim)
        self.conv_fc1 = nn.Conv1d(self.gru_dim * (1 + self.bidirectional), self.target_dim, kernel_size=1, padding=0)
        self.conv_fc2 = nn.Conv1d(self.target_dim, self.target_dim, kernel_size=1, padding=0)
        # torch.nn.init.normal_(self.conv_fc.weight, mean=0.2, std=0.01)

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
        
        xseq = torch.cat([xseq, xpairseq], dim=-1)
        if self.add_segment_info:
            xbpseq = xinputs["sequence_bp_segment"]
            xplseq = xinputs["sequence_pl_segment"]
            xseg = torch.cat([xbpseq, xplseq], -1)
            xseq = torch.cat([xseq, xseg], -1)

        xseq = self.seq_conv(xseq)
        xpl = self.struct_pl_conv(torch.cat([xpl, xstruct], dim=-1))

        bs, seq_len = xseq.size()[:2]
        xbpp_inp = xinputs["bpps"].permute(0, 3, 1, 2).contiguous()
        xbpp = self.bpp_conv(xbpp_inp)
        xbpp_mean = nn.AdaptiveAvgPool2d((seq_len, 1))(xbpp).squeeze(3).permute(0, 2, 1).contiguous()
        xbpp_max = nn.AdaptiveMaxPool2d((seq_len, 1))(xbpp).squeeze(3).permute(0, 2, 1).contiguous()
        xbpp_emb = torch.cat([xbpp_mean, xbpp_max], -1)
        xseqpl = xbpp_emb * xseq * xpl
        # xpl = xpl + xpl * xbpp_emb
        # bpps_raw_mean = torch.sum(xbpp_inp[:, 0, :, :], dim=2, keepdim=True)
        # bpps_raw_max = torch.max(xbpp_inp[:, 0, :, :], dim=2, keepdim=True).values

        # xbpp_mean = torch.sum(xbpp, dim=2).permute(0, 2, 1).contiguous()
        # xbpp_std = torch.mean(torch.std(xbpp, dim=1), dim=2).unsqueeze(2)
        # xbpp_max = torch.max(xbpp, dim=2).values.permute(0, 2, 1).contiguous()
        #  xbpp_pair = xbpp_mean.gather(xinputs["pair_sequence"], 1)
        xseq = torch.cat([xseq, xpl, xseqpl], -1)
        if self.add_entropy:
            # print(xinputs["sequence_entropy"].shape)
            xent = (xinputs["sequence_entropy"].unsqueeze(2) / 4.0 - 2.0)
            xent = self.ent_conv(xent)
            xseq = torch.cat([xseq, xent], -1)
        
        position = torch.clamp(torch.arange(0, seq_len, dtype=torch.float), 0, 10).to(xseq.device) / 10.0
        position = position.unsqueeze(0).unsqueeze(2).repeat(bs, 1, 1)
        xseq = torch.cat([xseq, position], -1)
        # xseq = self.pos(xseq.permute(1, 0, 2).contiguous()).permute(1, 0, 2).contiguous()
        # x = self.transformer(xtfm).permute(1, 0, 2).contiguous()
        # cont_emb = torch.cat([xbpp_mean, xbpp_max], dim=-1)
        # cont_emb = self.cont_fc(cont_emb)
        # x = torch.cat([xseq, xseqbpp, cont_emb], dim=-1)
        # x = self.pos(x)
        x, _ = self.gru(xseq)
        # x = self.pos(x.permute(1, 0, 2).contiguous()).permute(1, 0, 2).contiguous()
        # x = self.fc1(x)
        # x = self.transformer(x.permute(1, 0, 2).contiguous()).permute(1, 0, 2).contiguous()
        x = x[:, : self.max_seq_pred, :]
        x = self.drop(x)
        # x = self.conv_fc1(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        x = self.fc(x)
        return x


class RCNNGRUModelv7(ParamModel):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.hparams = hparams
        self.sequence_embedding = nn.Embedding(self.num_seq_tokens, self.seq_emb_dim)
        if self.use_6n:
            embeddings = np.load("data/seq7n_vectors.npy").astype('float32')
            self.sequence_embedding = nn.Embedding(len(embeddings), 128)
            self.sequence_embedding.weight = nn.Parameter(torch.from_numpy(embeddings))

        self.predicted_loop_embedding = nn.Embedding(self.num_pl_tokens, self.pl_emb_dim)
        # self.predicted_loop_embedding.weight = nn.Parameter(torch.from_numpy(np.load("data/pl_embed.npy")))

        if self.use_one_hot:
            self.seq_conv = Conv1dBnStack(self.num_seq_tokens, self.conv_channels)
            self.pl_conv = Conv1dBnStack(self.num_pl_tokens, self.conv_channels)
        else:
            seq_inp = self.seq_emb_dim * 1
            self.seq_conv = Conv1dBnStack(seq_inp, self.conv_channels, groups=1, kernel_size=5, padding=2)
            self.pl_conv = Conv1dBnStack(self.pl_emb_dim, self.conv_channels, groups=1, kernel_size=5, padding=2)

        bpp_conv_channels = [15] + self.bpp_conv_channels
        self.bpp_bnorm = nn.BatchNorm2d(15)
        self.bpp_conv = nn.Sequential(
            *[Conv2dBn(bpp_conv_channels[i], bpp_conv_channels[i + 1], kernel_size=1, padding=0) for i in range(len(bpp_conv_channels) - 1)]
        )
        inp_shape = (self.seq_emb_dim+self.pl_emb_dim)*1 + 1
        # self.pos = PositionalEncoding(inp_shape)
        self.transformer = nn.TransformerEncoder(
            CustomTransformerEncoderLayer(inp_shape, 4, 512, 0.1), 3)
        self.drop = nn.Dropout(self.spatial_dropout)
        self.fc = nn.Linear(inp_shape, self.target_dim)

    def forward(self, xinputs):
        if self.use_one_hot:
            xseq = onehot(xinputs["sequence"], self.num_seq_tokens)
            xpl = onehot(xinputs["predicted_loop_type"], self.num_pl_tokens)
        else:
            xseq = self.sequence_embedding(xinputs["sequence"])
            xpl = self.predicted_loop_embedding(xinputs["predicted_loop_type"])
        
        xbpp_inp = xinputs["bpps"][:, :, :, 0]
        # xseq_bpp = torch.bmm(xseq.permute(0, 2, 1).contiguous(), xbpp_inp).permute(0, 2, 1).contiguous()
        # xpl_bpp = torch.bmm(xpl.permute(0, 2, 1).contiguous(), xbpp_inp).permute(0, 2, 1).contiguous()

        x = torch.cat([xseq, xpl, torch.sum(xbpp_inp, dim=2, keepdim=True)], -1).permute(1, 0, 2).contiguous()
        print(x.shape)
        # x = self.pos(x)
        x = self.transformer(x).permute(1, 0, 2).contiguous()
        x = x[:, : self.max_seq_pred, :]
        x = self.drop(x)
        x = self.fc(x)
        return x


class SequenceEncoder(ParamModel):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.hparams = hparams
        conv_input = self.num_seq_tokens + self.num_pl_tokens + self.num_struct_tokens
        self.seq_conv = Conv1dBnStack(conv_input, self.conv_channels, groups=1, kernel_size=5, padding=2, drop=0.1)
        self.inp_shape = self.conv_channels[-1]
        self.pos1 = PositionalEncoding(self.inp_shape)
        self.pos2 = PositionalEncoding(128)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(self.inp_shape, 2, 256, 0.1), 2)
        self.decoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(128, 2, 256, 0.1), 2)
        self.sdrop = SpatialDropout(0.2)
        self.drop = nn.Dropout(0.1)
        self.fcs = nn.Linear(self.inp_shape, self.inp_shape)
        self.fc11 = nn.Linear(self.inp_shape, 128)
        self.fc12 = nn.Linear(self.inp_shape, 128)

        self.fc = nn.Linear(128, conv_input)

    def encode(self, x):
        # xpairseq = onehot(xinputs["pair_sequence"], self.num_seq_tokens)        
        x = self.seq_conv(x)
        # x = self.sdrop(x)
        x = x.permute(1, 0, 2).contiguous()
        x = self.pos1(x)
        x = self.transformer(x).permute(1, 0, 2).contiguous()
        # x = self.sdrop(x)
        x1 = self.fcs(x)
        x1 = torch.softmax(x1, dim=1)
        xx = torch.sum(x * x1, dim=1)
        # xx = torch.mean(x, dim=1)
        mu_z = self.fc11(xx)

        logvar_z = self.fc12(xx)
        return x, mu_z, logvar_z

    def reparameterize(self, mu: Variable, logvar: Variable) -> Variable:
        if self.training:
            # multiply log variance with 0.5, then in-place exponent
            # yielding the standard deviation
            sample_z = []
            for _ in range(10):
                std = logvar.mul(0.5).exp_()  # type: Variable
                eps = Variable(std.data.new(std.size()).normal_())
                sample_z.append(eps.mul(std).add_(mu))

            return sample_z
        else:
            # During inference, we simply spit out the mean of the
            # learned distribution for the current input.  We could
            # use a random sample from the distribution, but mu of
            # course has the highest probability.
            return mu

    def decode(self, z, seq_len):
        # bs, seq_len = x.size()[:2]
        z = z.unsqueeze(1).repeat(1, seq_len, 1)
        # x = torch.cat([])
        z = z.permute(1, 0, 2).contiguous()
        z = self.pos2(z)
        z = self.decoder(z)
        z = z.permute(1, 0, 2).contiguous()
        z = self.fc(z)
        return z

    def forward(self, x: Variable) -> (Variable, Variable, Variable):
        xx, mu, logvar = self.encode(x)
        bs, seq_len = x.size()[:2]
        # print(mu.shape, logvar.shape)
        z = self.reparameterize(mu, logvar)
        # print(z[0].shape)
        # print(self.decode(z[0]).shape)
        if self.training:
            return [self.decode(z, seq_len) for z in z], mu, logvar
        else:
            return self.decode(z, seq_len), mu, logvar
        # return self.decode(z), mu, logvar


class PretrainedTransformer(ParamModel):
    def __init__(self, hparams, model):
        super().__init__(hparams)
        self.hparams = hparams
        self.seq_encoder = model
        bpp_conv_channels = [7] + self.bpp_conv_channels
        self.bpp_bnorm = nn.BatchNorm2d(15)
        self.bpp_encoder = nn.Sequential(
            *[Conv2dBn(bpp_conv_channels[i], bpp_conv_channels[i + 1], kernel_size=1, padding=0) for i in range(len(bpp_conv_channels) - 1)]
        )
        inp_shape = 128 + self.conv_channels[-1] + 17*2 + self.bpp_conv_channels[-1] * 2
        self.drop = nn.Dropout(0.0)
        self.sig = nn.Sigmoid()
        self.pos = PositionalEncoding(inp_shape)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(inp_shape, 2, 256, 0.1), 2)
        # self.transformer = nn.LSTM(inp_shape, inp_shape, bidirectional=True, num_layers=2, dropout=0.1)
        self.gru = nn.LSTM(
            input_size=inp_shape,
            hidden_size=self.gru_dim,
            bidirectional=self.bidirectional,
            batch_first=True,
            num_layers=self.gru_layers,
            dropout=self.dropout_prob,
        )
        self.fc = nn.Linear(self.gru_dim * (1 + self.bidirectional), self.target_dim)

    def get_bpp_features(self, x, seq_len):
        xbpp_conv = self.bpp_encoder(x[:, :, :, :7].permute(0, 3, 1, 2).contiguous())
        xbpp_mean = nn.AdaptiveAvgPool2d((seq_len, 1))(xbpp_conv).squeeze(3).permute(0, 2, 1).contiguous()
        xbpp_max = nn.AdaptiveMaxPool2d((seq_len, 1))(xbpp_conv).squeeze(3).permute(0, 2, 1).contiguous()
        xbpp_emb = torch.cat([xbpp_mean, xbpp_max], -1)
        return xbpp_emb

    def forward(self, xinputs):
        xseq = onehot(xinputs["sequence"], self.num_seq_tokens)
        xstruct = onehot(xinputs["structure"], self.num_struct_tokens)
        xpl = onehot(xinputs["predicted_loop_type"], self.num_pl_tokens)
        x = torch.cat([xseq, xstruct, xpl], -1)
        # self.bpp_model.to(xinputs["sequence"].device)
        bs, seq_len = x.size()[:2]
        xx, z, _ = self.seq_encoder.encode(x)
        z2 = self.seq_encoder.decode(z, seq_len)

        z = z.unsqueeze(1).repeat(1, seq_len, 1)
        xbpp = xinputs["bpps"]
        xbpp_emb = self.get_bpp_features(xbpp, seq_len)
        # b = torch.ones(bs, seq_len, 1, device=xseq.device)
        x = torch.cat([x, xbpp_emb, z, z2, xx], -1)
        #x = x.permute(1, 0, 2).contiguous()
        #x = self.pos(x)
        #x = self.transformer(x).contiguous()
        #x = x.permute(1, 0, 2).contiguous()
        x, _ = self.gru(x)
        x = x[:, : self.max_seq_pred, :]
        x = self.fc(self.drop(x))
        return x
