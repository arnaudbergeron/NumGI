from __future__ import annotations

import torch

from NumGI.Model.Inference import batch_inference
from NumGI.Model.Model import TransformerNet


def test_batch_inference():
    """Test TransformerNet class."""
    _max_src_len = 2
    model = TransformerNet(
        num_src_vocab=4,
        num_tgt_vocab=4,
        embedding_dim=8,
        hidden_size=8,
        nheads=1,
        n_layers=1,
        max_src_len=_max_src_len,
        max_tgt_len=_max_src_len - 1,
        dropout=0.1,
    )
    fake_input = torch.tensor([[3, 3], [3, 3]])

    assert batch_inference(fake_input, model, {"START": 0, "PAD": 1, "END": 2}) is not None
