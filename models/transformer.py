import math
import torch
from torch import nn

class PositionalEncoder(nn.Module):

    def __init__(self, d_pe, max_len = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_pe, 2) * (-math.log(10000.0) / d_pe))
        pe = torch.zeros(max_len, 1, d_pe)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        return torch.cat((x, self.pe[:x.size(0)]), dim=2)

class Transformer(nn.Module):
    def __init__(self, d_model, d_pe, nhead):
        super().__init__()
        self.positional_encoder = PositionalEncoder(d_pe)
        d_encoded = d_model + d_pe
        self.transformer = nn.Transformer(
            d_model=d_encoded,
            nhead=nhead,
        )
        self.decoder = nn.Linear(d_encoded, d_model)

    def forward(self, src, tgt, src_padding_mask = None, tgt_padding_mask = None):
        src_pos_encoding = self.positional_encoder(src)
        tgt_pos_encoding = self.positional_encoder(tgt)
        encoded = self.transformer(
            src_pos_encoding,
            tgt_pos_encoding,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
        )
        return self.decoder(encoded)
