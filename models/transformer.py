import math
import torch
from torch import nn

class PositionalEncoder(nn.Module):
    """
        From official implementation: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        # Concatenate instead of add to allow use with low-dimensional x
        return torch.cat((x, self.pe[:x.size(0)]), dim=2)
        # return x + self.pe[:x.size(0)]

class Transformer(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.positional_encoder = PositionalEncoder(d_model)
        positional_encoder_d = 2 * d_model
        self.transformer = nn.Transformer(
            d_model=positional_encoder_d,
            nhead=nhead,
        )
        # encoder_layer = nn.TransformerEncoderLayer(positional_encoder_d, nhead)
        # self.encoder = nn.TransformerEncoder(encoder_layer, 6)
        self.decoder = nn.Linear(positional_encoder_d, d_model)

    def forward(self, src, tgt):
        # return self.transformer(self.positional_encoder(src), tgt)
        src_pos_encoding = self.positional_encoder(src)
        tgt_pos_encoding = self.positional_encoder(tgt)
        # encoded = self.encoder(positional_encoding)
        # return self.decoder(encoded)
        encoded = self.transformer(src_pos_encoding, tgt_pos_encoding)
        return self.decoder(encoded)
