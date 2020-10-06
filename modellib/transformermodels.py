import math

import numpy as np
import torch
from torch import nn


class ConvBn(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, dropout, conv_type, activation=True):

        super(ConvBn, self).__init__()
        self.conv_type = conv_type
        if self.conv_type == "1D":
            self.conv = nn.Conv1d(in_dim, out_dim, kernel_size=kernel_size, padding=kernel_size // 2)
            self.bn = nn.BatchNorm1d(out_dim)
        elif self.conv_type == "2D":
            self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, padding=kernel_size // 2)
            self.bn = nn.BatchNorm2d(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activation:
            x = nn.GELU()(x)
        x = self.dropout(x)
        return x


class Conv2D1x1(nn.Module):
    def __init__(self, input_dim, channel_list, dropout):
        super().__init__()
        channel_list = [input_dim] + channel_list
        self.conv_layers = []
        for i, (a, b) in enumerate(zip(channel_list[:-1], channel_list[1:])):
            if i == len(channel_list) - 1:
                layer = ConvBn(a, b, 1, dropout, "2D", activation=False)
            else:
                layer = ConvBn(a, b, 1, dropout, "2D")
            self.conv_layers.append(layer)
        self.conv_layers = nn.ModuleList(self.conv_layers)

    def forward(self, x):
        for i, layer in enumerate(self.conv_layers):
            x_ = layer(x)
            x = x_
        return x


class Conv1Dlayer(nn.Module):
    def __init__(self, input_dim, channel_list, kernel_size_list, dropout):
        super().__init__()
        channel_list = [input_dim] + hidden_channel_list
        self.conv_layers = nn.ModuleList(
            [ConvBn(a, b, k, dropout, "1D") for a, b, k in zip(channel_list[:-1], channel_list[1:], kernel_size_list)]
        )

    def forward(self, x):
        for i, layer in enumerate(self.conv_layers):
            x_ = layer(x)
            x = x_
        return x


class SingleHeadStaticAttn(nn.Module):
    def __init__(self, input_feats, ff_dim, ff_dropout):
        super().__init__()
        self.input_feats = input_feats
        self.norm_layer1 = nn.LayerNorm(input_feats)
        self.norm_layer2 = nn.LayerNorm(input_feats)
        self.dp = nn.Dropout(ff_dropout)

        self.ff = nn.Sequential(*[nn.Linear(input_feats, ff_dim), nn.GELU(), nn.Linear(ff_dim, input_feats)])

    def forward(self, x, attn_map):
        """
        shape of x: batch*num_feats/channel*seq_len : output of the conv layer can be directly fed into this
        """
        xl = self.norm_layer1(x)
        # attn_map = torch.softmax(attn_map, dim=1)
        x_ = torch.bmm(attn_map, xl.permute(1, 0, 2).contiguous()).permute(1, 0, 2).contiguous()
        x = x + self.dp(x_)
        xl = self.norm_layer2(x)
        x_ = self.ff(xl)
        x = x + self.dp(x_)
        return x


class MultiHeadStaticAttn(nn.Module):
    def __init__(self, singlehead, input_dim, conv_channel_list, conv_dropout):
        super().__init__()
        self.attn_conv = Conv2D1x1(input_dim, conv_channel_list, dropout=conv_dropout)
        self.fc = nn.Linear(conv_channel_list[-1] * singlehead.input_feats, singlehead.input_feats)
        self.singlehead = singlehead

    def forward(self, x, fixed_attns):
        """
        shape of x: batch*num_feats/channel*seq_len : output of the conv layer can be directly fed into this
        fixes_attns: batch*num_heads*seq_len*seq_len
        """
        fixed_attns = self.attn_conv(fixed_attns)
        num_heads = fixed_attns.size(1)
        heads = []
        for i in range(num_heads):
            head = self.singlehead(x, fixed_attns[:, i, :, :])
            heads.append(head)
        multihead_concat = torch.cat(heads, -1)
        return self.fc(multihead_concat)


class TransformerCustomEncoder(nn.Module):
    def __init__(self, num_layers, input_feats, ff_dim, attn_input_dim, attn_channel_list, ff_dropout, attn_dropout):
        super().__init__()
        layers = []
        for i in range(num_layers):
            if i == 0:
                singlehead = SingleHeadStaticAttn(input_feats, ff_dim, ff_dropout)
                multihead = MultiHeadStaticAttn(singlehead, attn_input_dim, attn_channel_list, attn_dropout)
                layers.append(multihead)
            else:
                mh = nn.TransformerEncoderLayer(input_feats, 4, ff_dim, ff_dropout, activation="gelu")
                layers.append(mh)
        self.encoder = nn.ModuleList(layers)

    def forward(self, x, attns):
        for i, layer in enumerate(self.encoder):
            if i == 0:
                x = layer(x, attns)
            else:
                x = layer(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=2000):
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


class CustomTransformerEncoderLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(CustomTransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, src, src_mask = None, src_key_padding_mask = None):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src = self.norm1(src)
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        # src = self.norm2(src)
        return src