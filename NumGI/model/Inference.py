from __future__ import annotations

import torch


def batch_inference(
    input_seqs: torch.tensor, model: torch.nn, start_id: int, pad_id: int, end_id: int
) -> torch.tensor:
    """Performs Batch Inference on a list of input sequences.

    Args:
        input_seqs (torch.tensor): Input Equation(s) to be solved should be < batch_size
        model (torch.nn): Trained Transformer Model
        start_id (int): Start of Sequence ID token
        pad_id (int): Padding ID token
        end_id (int): End of Sequence ID token

    Returns:
        torch.tensor: Output Solution(s) to solved by the model
    """
    model = model.to("cpu")
    model.tgt_mask = model.tgt_mask.to("cpu")
    model.eval()
    with torch.no_grad():
        mask_in = input_seqs == pad_id
        out = torch.full(input_seqs.shape, pad_id)
        out[:][0] = start_id
        for i in range(1, out.shape[1]):
            mask_out = out == pad_id
            output = model(input_seqs, out, mask_in, mask_out)
            output = output.exp()
            out[:][i] = torch.multinomial(output[i - 1, :, :], 1, replacement=False)

            if torch.sum(out[:][i] != end_id) == 0:
                break

    return out
