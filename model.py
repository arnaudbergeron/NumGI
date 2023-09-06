import torch
import torch.nn as nn

class TransformerNet(nn.Module):
    """This is the main Transformer model we will be using for the project.

    Args:
        nn (_type_): _description_
    """
    def __init__(self, num_src_vocab, num_tgt_vocab, embedding_dim, hidden_size, nheads, n_layers, max_src_len, max_tgt_len, dropout):
        super(TransformerNet, self).__init__()
        # embedding layers
        self.enc_embedding = nn.Embedding(num_src_vocab, embedding_dim)
        self.dec_embedding = nn.Embedding(num_tgt_vocab, embedding_dim)

        # positional encoding layers
        self.enc_pe = PositionalEncoding(embedding_dim, max_len = max_src_len)
        self.dec_pe = PositionalEncoding(embedding_dim, max_len = max_tgt_len)

        # encoder/decoder layers
        enc_layer = nn.TransformerEncoderLayer(embedding_dim, nheads, hidden_size, dropout)
        dec_layer = nn.TransformerDecoderLayer(embedding_dim, nheads, hidden_size, dropout)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers = n_layers)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers = n_layers)

        # final dense layer
        self.dense = nn.Linear(embedding_dim, num_tgt_vocab)

        self.tgt_mask = self.src_att_mask(max_tgt_len)

    def src_att_mask(self, src_len):
        mask = (torch.triu(torch.ones(src_len, src_len)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, tgt, mask_in, mask_out):
        src, tgt = self.enc_embedding(src).permute(1, 0, 2), self.dec_embedding(tgt).permute(1, 0, 2)
        src, tgt = self.enc_pe(src), self.dec_pe(tgt)
        memory = self.encoder(src, src_key_padding_mask=mask_in)

        # tgt_mask = self.src_att_mask(tgt.shape[0]).to(tgt.device)

        transformer_out = self.decoder(tgt, memory,tgt_mask=self.tgt_mask, tgt_key_padding_mask=mask_out)
        final_out = self.dense(transformer_out)
        return final_out
  

class PositionalEncoding(nn.Module):
    """This is the Positional Encoding layer for the Transformer model.ow

    Args:
        nn (_type_): _description_
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)