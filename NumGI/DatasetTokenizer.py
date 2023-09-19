from __future__ import annotations

import random

from NumGI.EquationTokenizer import defaultTokenizer
from NumGI.EquationTokenizer import EquationTokenizer


class DatasetTokenizer(EquationTokenizer):
    """Class for tokenizing a dataset."""

    def __init__(self, x: list, y: list, useDefaultTokenizer=False, isSympy=False):
        """Creates a DatasetTokenizer object. Which holds your tokenizer and data.

        Args:
            x (list): Either a list of sympy equations or a list of sympy_to_list equations.
            y (list): Either a list of sympy solutions or a list of sympy_to_list solutions
            useDefaultTokenizer (bool, optional): If you want to use the default tokenizer.
              Normally used when creating a Dataset that you wish to save.
              Defaults to False.
        """
        super().__init__()

        # shuffle lists
        temp = list(zip(x, y))
        random.shuffle(temp)
        x, y = zip(*temp)
        x, y = list(x), list(y)

        if isinstance(x[0], list):
            self.x = x
            self.y = y

        else:
            self.x = [self.sympy_to_list(i) for i in x]
            self.y = [self.sympy_to_list(i) for i in y]

        if useDefaultTokenizer:
            print("Using default tokenizer.")
            self.tokenize_dict, self.decode_dict, self.tokenize, self.decode = defaultTokenizer()
            self.dict_size = len(self.tokenize_dict)
            self.char_set = set(self.tokenize_dict.keys())

        else:
            self.char_set = self.create_set_char()
            self.create_tokenizer(self.char_set)

        self.x_tokenized = [self.tokenize(i) for i in self.x]
        self.y_tokenized = [self.tokenize(i) for i in self.y]

        self.x_tokenized = self.tensorize_and_pad(self.x_tokenized)
        self.max_length = self.x_tokenized.shape[1]
        self.y_tokenized = self.tensorize_and_pad_by_len(self.y_tokenized, self.max_length)

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
