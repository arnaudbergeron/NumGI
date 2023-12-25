from __future__ import annotations

import torch

from NumGI.DatasetTokenizer import DatasetTokenizer


class LoadTokenizer(DatasetTokenizer):
    """The tokenizer used when loading data from files."""

    def __init__(self, x_files, y_files):
        default_tokenized_x = []
        default_tokenized_y = []

        temp_data = [["1", "2"]]
        tempTokenizer = DatasetTokenizer(
            temp_data, temp_data, useDefaultTokenizer=True, isSympy=False
        )

        # load files
        max_length = 0
        for x_file, y_file in zip(x_files, y_files):
            _torch_x = torch.load(x_file)
            _torch_y = torch.load(y_file)
            default_tokenized_x.append(_torch_x)
            default_tokenized_y.append(_torch_y)
            max_length = max(max_length, _torch_x.shape[1])
            max_length = max(max_length, _torch_y.shape[1])

        for idx, (x, y) in enumerate(zip(default_tokenized_x, default_tokenized_y)):
            default_tokenized_x[idx] = tempTokenizer.pad_by_len(x, max_length)
            default_tokenized_y[idx] = tempTokenizer.pad_by_len(y, max_length)

        default_combined_x_torch = torch.cat(default_tokenized_x, axis=0)
        default_combined_y_torch = torch.cat(default_tokenized_y, axis=0)

        new_x = [tempTokenizer.tokens_to_list(i) for i in default_combined_x_torch.tolist()]
        new_y = [tempTokenizer.tokens_to_list(i) for i in default_combined_y_torch.tolist()]

        super().__init__(new_x, new_y, useDefaultTokenizer=False, isSympy=False)
