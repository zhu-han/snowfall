#!/usr/bin/env python3

# Copyright (c)  2021  University of Chinese Academy of Sciences (author: Han Zhu)
# Apache 2.0

import k2
import math
import torch
from torch import Tensor, nn
from typing import Dict, List, Optional, Tuple

from snowfall.common import get_texts
from snowfall.models import AcousticModel


class Transformer(AcousticModel):
    """
    Args:
        num_features (int): Number of input features
        num_classes (int): Number of output classes
        subsampling_factor (int): subsampling factor of encoder (the convolution layers before transformers)
        d_model (int): attention dimension
        nhead (int): number of head
        dim_feedforward (int): feedforward dimention
        num_layers (int): number of transformer layers
        dropout (float): dropout rate
        normalize_before (bool): whether to use layer_norm before the first block.
    """

    def __init__(self, num_features: int, num_classes: int, subsampling_factor: int = 4,
                 d_model: int = 256, nhead: int = 4, dim_feedforward: int = 2048,
                 num_layers: int = 12, dropout: float = 0.1, normalize_before: bool = True) -> None:
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.subsampling_factor = subsampling_factor
        if subsampling_factor != 4:
            raise NotImplementedError("Support only 'subsampling_factor=4'.")

        self.encoder_embed = Conv2dSubsampling(num_features, d_model)
        self.encoder_pos = PositionalEncoding(d_model, dropout)

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)

        if normalize_before:
            encoder_norm = nn.LayerNorm(d_model)
        else:
            encoder_norm = None

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers, encoder_norm)

        self.encoder_output_layer = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x: Tensor, supervisions: Optional[Dict] = None) -> Tensor:
        """
        Args:
            x: Tensor of dimension (batch_size, num_features, input_length).
            supervisions : Supervison in lhotse format, i.e., batch['supervisions']

        Returns:
            Tensor: After log-softmax tensor of dimension (batch_size, number_of_classes, input_length).
        """
        x = x.permute(0, 2, 1)  # (B, F, T) -> (B, T, F)

        x = self.encoder_embed(x)
        x = self.encoder_pos(x)
        x = x.permute(1, 0, 2)  # (B, T, F) -> (T, B, F)

        mask = encoder_padding_mask(x.size(0), supervisions)
        mask = mask.to(x.device) if mask != None else None

        x = self.encoder(x, src_key_padding_mask=mask)  # (T, B, F)
        x = self.encoder_output_layer(x).permute(1, 2, 0)  # (T, B, F) ->(B, F, T)
        x = nn.functional.log_softmax(x, dim=1)  # (B, F, T)

        return x


class TransformerEncoderLayer(nn.Module):
    """
    Modified from torch.nn.TransformerEncoderLayer. Add support of normalize_before, 
    i.e., use layer_norm before the first block.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
        normalize_before: whether to use layer_norm before the first block.

    Examples::
        >>> encoder_layer = TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = "relu", normalize_before: bool = True) -> None:
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=0.0)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        self.normalize_before = normalize_before

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = nn.functional.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """
        Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            src: (S, N, E).
            src_mask: (S, S).
            src_key_padding_mask: (N, S).
            S is the source sequence length, T is the target sequence length, N is the batch size, E is the feature number
        """
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = residual + self.dropout1(src2)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src2)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


def _get_activation_fn(activation: str):
    if activation == "relu":
        return nn.functional.relu
    elif activation == "gelu":
        return nn.functional.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class Conv2dSubsampling(nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).
        Modified from https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/transformer/subsampling.py

    Args:
        idim: Input dimension.
        odim: Output dimension.

    """

    def __init__(self, idim: int, odim: int) -> None:
        """Construct an Conv2dSubsampling object."""
        super(Conv2dSubsampling, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=odim, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=odim, out_channels=odim, kernel_size=3, stride=2),
            nn.ReLU(),
        )
        self.out = nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim)

    def forward(self, x: Tensor) -> Tensor:
        """Subsample x.

        Args:
            x: Input tensor of dimension (batch_size, input_length, num_features). (#batch, time, idim).

        Returns:
            torch.Tensor: Subsampled tensor of dimension (batch_size, input_length, d_model).
                where time' = time // 4.

        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        return x


class PositionalEncoding(nn.Module):
    """
    Positional encoding.

    Args:
        d_model: Embedding dimension.
        dropout: Dropout rate.
        max_len: Maximum input length.

    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        """Construct an PositionalEncoding object."""
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x: Tensor) -> None:
        """Reset the positional encodings."""
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1):
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        pe = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: Tensor) -> Tensor:
        """
        Add positional encoding.

        Args:
            x: Input tensor of dimention (batch_size, input_length, d_model).

        Returns:
            torch.Tensor: Encoded tensor of dimention (batch_size, input_length, d_model).

        """
        self.extend_pe(x)
        x = x * self.xscale + self.pe[:, : x.size(1)]
        return self.dropout(x)


class Noam(object):
    """
    Implements Noam optimizer. Proposed in "Attention Is All You Need", https://arxiv.org/pdf/1706.03762.pdf
    Modified from https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/transformer/optimizer.py
    
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        model_size: attention dimension of the transformer model
        factor: learning rate factor
        warm_step: warmup steps
    """

    def __init__(self, params, model_size: int = 256, factor: float = 10.0, warm_step: int = 25000) -> None:
        """Construct an Noam object."""
        self.optimizer = torch.optim.Adam(params, lr=0, betas=(0.9, 0.98), eps=1e-9)
        self._step = 0
        self.warmup = warm_step
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    @property
    def param_groups(self):
        """Return param_groups."""
        return self.optimizer.param_groups

    def step(self):
        """Update parameters and rate."""
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        """Implement `lrate` above."""
        if step is None:
            step = self._step
        return (
                self.factor
                * self.model_size ** (-0.5)
                * min(step ** (-0.5), step * self.warmup ** (-1.5))
        )

    def zero_grad(self):
        """Reset gradient."""
        self.optimizer.zero_grad()

    def state_dict(self):
        """Return state_dict."""
        return {
            "_step": self._step,
            "warmup": self.warmup,
            "factor": self.factor,
            "model_size": self.model_size,
            "_rate": self._rate,
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict):
        """Load state_dict."""
        for key, value in state_dict.items():
            if key == "optimizer":
                self.optimizer.load_state_dict(state_dict["optimizer"])
            else:
                setattr(self, key, value)


def encoder_padding_mask(max_len: int, supervisions: Optional[Dict] = None) -> Optional[Tensor]:
    """Make mask tensor containing indices of padded part.

    Args:
        max_len: maximum length of input features
        supervisions : Supervison in lhotse format, i.e., batch['supervisions']

    Returns:
        Tensor: Mask tensor of dimension (batch_size, input_length), True denote the masked indices.
    """
    if supervisions == None:
        return None
    
    supervision_segments = torch.stack(
        (supervisions['sequence_idx'],
         supervisions['start_frame'],
         supervisions['num_frames']), 1).to(torch.int32)

    lengths = [0 for _ in range(int(max(supervision_segments[:, 0])) + 1)]
    for sequence_idx, start_frame, num_frames in supervision_segments:
        lengths[sequence_idx] = start_frame + num_frames
    
    lengths = [((i -1) // 2 - 1) // 2 for i in lengths]
    bs = int(len(lengths))
    seq_range = torch.arange(0, max_len, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, max_len)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand

    return mask