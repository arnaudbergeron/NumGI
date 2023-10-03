from __future__ import annotations

import random

from NumGI.EquationTokenizer import EquationTokenizer


class DatasetTokenizer(EquationTokenizer):
    """
    A class for tokenizing datasets.

    Args:
        x (list): A list of input data.
        y (list): A list of output data.
        useDefaultTokenizer (bool, optional): Whether to use the default tokenizer.
          Defaults to False.
        isSympy (bool, optional): Whether the input data is in SymPy format or sympy-list format.
          Defaults to True.
    """

    def __init__(self, x, y, useDefaultTokenizer=False, isSympy=True):
        super().__init__(useDefaultTokenizer=useDefaultTokenizer)

        # shuffle lists
        temp = list(zip(x, y))
        random.shuffle(temp)
        x, y = zip(*temp)
        x, y = list(x), list(y)

        if isSympy:
            self.x = [self.sympy_to_list(i) for i in x]
            self.y = [self.sympy_to_list(i) for i in y]

        else:
            self.x = x
            self.y = y

        if not useDefaultTokenizer:
            self.char_set = self.create_set_char()
            self.create_tokenizer(self.char_set)

        self.x_tokenized = [self.tokenize(i) for i in self.x]
        self.y_tokenized = [self.tokenize(i) for i in self.y]

        self.x_tokenized = self.tensorize_and_pad(self.x_tokenized)
        self.max_length_x = self.x_tokenized.shape[1]

        self.y_tokenized = self.tensorize_and_pad(self.y_tokenized)
        self.max_length_y = self.y_tokenized.shape[1]

    def create_set_char(self):
        char_list_x = [j for i in self.x for j in i]
        char_list_y = [j for i in self.y for j in i]
        char_set = set(char_list_x).union(set(char_list_y))
        return char_set

    def sympy_to_padded_tokens(self, eq):
        """Takes in a sympy equation and outputs a tokenized padded list."""
        seq = self.sympy_to_list(eq)

        if set(seq) - self.char_set != set():
            raise Exception(
                f"The equation contains characters not in the character set. \
                    The models output will be non-sensical. \
                    Remove characters: {(set(seq) - self.char_set)}"
            )

        seq = self.tokenize(seq)
        seq = self.tensorize_and_pad_by_len([seq], self.max_length)
        return seq

    def split(self, factor):
        split_n = int(self.x_tokenized.shape[0] * factor)

        self.x_train = self.x_tokenized[:split_n]
        self.y_train = self.y_tokenized[:split_n]
        self.x_val = self.x_tokenized[split_n:]
        self.y_val = self.y_tokenized[split_n:]
