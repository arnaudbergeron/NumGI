from __future__ import annotations

import random

import sympy as sp
import torch

from NumGI.DatasetTokenizer import DatasetTokenizer


class LossDataset(DatasetTokenizer):
    """Docstring for LossDataset.

    Args:
        DatasetTokenizer (DatasetTokenizer): DatasetTokenizer to create loss dataset from.
    """

    def __init__(self, eq_dataset: DatasetTokenizer, N: int, ell_norm: int = 1):
        self.eq_dataset = eq_dataset
        self.var_dict = self.create_var_dict()
        self.loss = self.calculate_n_pairwise_loss(N, ell_norm)

    def create_var_dict(self):
        var_dict = {}
        equations = self.eq_dataset.y_tokenized.tolist()
        for i, eq in enumerate(equations):
            solution = self.eq_dataset.tokens_to_sympy(eq)
            if frozenset(solution.free_symbols) not in var_dict:
                var_dict[frozenset(solution.free_symbols)] = [[solution, i]]
            else:
                var_dict[frozenset(solution.free_symbols)].append([solution, i])
        return var_dict

    def calculate_n_pairwise_loss(self, N, ell_norm):
        loss = torch.zeros((3, N))
        possible_symbols = self.var_dict.keys()

        first_batch = int(0.9 * N)
        second_batch = N - first_batch
        for i in range(first_batch):
            chosen_symbols = random.choice(list(possible_symbols))

            sol_sympy_1 = random.choice(self.var_dict[chosen_symbols])
            sol_sympy_2 = random.choice(self.var_dict[chosen_symbols])
            integral = sp.Abs(sol_sympy_1[0].rhs - sol_sympy_2[0].rhs) ** ell_norm
            for symbol in chosen_symbols:
                integral = sp.integrate(integral, (symbol, -sp.oo, sp.oo))

            loss[0, i] = sol_sympy_1[1]
            loss[1, i] = sol_sympy_2[1]
            if integral.is_number:
                loss[2, i] = float(integral)
            else:
                loss[2, i] = torch.inf

        for i in range(second_batch):
            chosen_symbols = random.sample(possible_symbols, 2)
            sol_sympy_1 = random.choice(self.var_dict[chosen_symbols[0]])
            sol_sympy_2 = random.choice(self.var_dict[chosen_symbols[1]])

            loss[0, i] = sol_sympy_1[1]
            loss[1, i] = sol_sympy_2[1]
            loss[2, i] = torch.inf

        return loss
