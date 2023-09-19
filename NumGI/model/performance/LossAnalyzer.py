from __future__ import annotations

import pandas as pd
import sympy as sp
import torch
import torch.nn as nn

from NumGI.DatasetTokenizer import DatasetTokenizer


class LossAnalyzer:
    """Used to analyze the loss of the model with respect to multiples equation parameters."""

    def __init__(
        self,
        input_tensor: torch.tensor,
        true_output_tensor: torch.tensor,
        infered_output_tensor: torch.tensor,
        tokenizer: DatasetTokenizer,
    ):
        self.input_tensor = input_tensor
        self.true_tensor = true_output_tensor
        self.infered_tensor = infered_output_tensor
        self.tokenizer = tokenizer

        self.input_sympy = self.tokenizer.tensor_to_sympy(self.input_tensor)
        self.true_sympy = self.tokenizer.tensor_to_sympy(self.true_tensor)
        self.infered_sympy = self.tokenizer.tensor_to_sympy(self.infered_tensor)

        self.get_loss()
        self.get_sympy_metrics()

    def get_loss(self):
        """Calculates all the various losses for all equations."""
        self.CE_loss = self.get_CE_loss()
        self.discrete_loss = self.get_discrete_loss()

        self.input_length = self.get_equation_length(self.input_tensor)
        self.true_length = self.get_equation_length(self.true_tensor)
        self.infered_length = self.get_equation_length(self.infered_tensor)

        self.length_diff = torch.abs(self.true_length - self.infered_length)

    def get_sympy_metrics(self):
        """Calculates all the various sympy metrics for all equations."""
        self.equation_order = self.get_eq_order()

    def get_CE_loss(self):
        """Returns the Cross Entropy loss of the inference data."""
        loss = nn.CrossEntropyLoss(reduce=False)

        return loss(self.infered_tensor.float(), self.true_tensor.float())

    def get_discrete_loss(self):
        """Returns the discrete metric loss of the inference data."""
        return torch.sum(self.infered_tensor != self.true_tensor, axis=1)

    def get_equation_length(self, eq):
        """Returns the length of the input data."""
        pad_id = self.tokenizer.tokenize_dict["PAD"]

        return torch.sum(eq != pad_id, axis=1)

    def get_eq_order(self):
        """Returns the order of each equation."""
        orders = torch.zeros(self.input_tensor.shape[0])

        for i in range(len(self.input_sympy)):
            orders[i] = sp.ode_order(self.input_sympy[i], self.true_sympy[i].lhs)

        return orders

    def create_dataframe(self):
        """Creates a dataframe with each row being an equation and each column being a metric."""
        df = pd.DataFrame()

        df["equation"] = self.input_sympy
        df["true_solution"] = self.true_sympy
        df["infered_solution"] = self.infered_sympy

        df["CE_loss"] = self.CE_loss
        df["discrete_loss"] = self.discrete_loss
        df["input_length"] = self.input_length
        df["true_length"] = self.true_length
        df["infered_length"] = self.infered_length
        df["length_diff"] = self.length_diff
        df["equation_order"] = self.equation_order

        return df
