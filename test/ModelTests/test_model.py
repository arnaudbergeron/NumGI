from __future__ import annotations

import torch

from NumGI.model.Model import TransformerNet


def test_transformer_net():
    """Test TransformerNet class."""
    _max_src_len = 2
    model = TransformerNet(
        num_src_vocab=1,
        num_tgt_vocab=1,
        embedding_dim=8,
        hidden_size=8,
        nheads=1,
        n_layers=1,
        max_src_len=_max_src_len,
        max_tgt_len=_max_src_len - 1,
        dropout=0.1,
    )
    fake_input = torch.tensor([[0, 0]])
    fake_output = torch.tensor([[0]])

    mask_in = fake_input == 1
    mask_out = fake_output == 1

    assert model is not None
    assert model.enc_embedding is not None
    assert model.dec_embedding is not None
    assert model.enc_pe is not None
    assert model.dec_pe is not None
    assert model.encoder is not None
    assert model.decoder is not None
    assert model.dense is not None
    assert model.tgt_mask is not None
    assert model.forward(fake_input, fake_output, mask_in, mask_out) is not None
