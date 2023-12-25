from __future__ import annotations

import random

import sympy as sp
import torch

from NumGI.DatasetTokenizer import DatasetTokenizer


class LossDataset:
    """Docstring for LossDataset.

    Args:
        DatasetTokenizer (DatasetTokenizer): DatasetTokenizer to create loss dataset from.
    """

    def __init__(self, eq_dataset: DatasetTokenizer, N: int, ell_norm: int = 1):
        self.eq_dataset = eq_dataset
        self.grid_size = (100, 100, 1000)
        self.var_dict = self.create_var_dict()
        self.loss = self.calculate_n_pairwise_loss(N, ell_norm)
        self.max_integral_value = 10e10  # we can play with this value

    def create_var_dict(self):
        """Creates a dictionary of different variables and their corresponding equations.

        Returns:
            _type_: _description_
        """
        var_dict = {}
        equations = self.eq_dataset.y_tokenized.tolist()
        self.solutions = []
        for i, eq in enumerate(equations):
            sol = self.eq_dataset.tokens_to_sympy(eq)
            self.solutions.append(sol)
            if frozenset(sol.free_symbols) not in var_dict:
                var_dict[frozenset(sol.free_symbols)] = [[sol, i]]
            else:
                var_dict[frozenset(sol.free_symbols)].append([sol, i])
        return var_dict

    def calculate_n_pairwise_loss(self, N, ell_norm):
        loss = torch.zeros((3, N))
        possible_symbols = self.var_dict.keys()

        possible_symbols = [i for i in possible_symbols if len(self.var_dict[i]) > 1]

        first_batch = int(0.9 * N)
        second_batch = N - first_batch
        for i in range(first_batch):
            print(i)
            chosen_symbols = random.choice(list(possible_symbols))

            possible_equations = {i[1] for i in self.var_dict[chosen_symbols]}

            idx_sympy_1, idx_sympy_2 = random.sample(possible_equations, 2)
            sol_sympy_1 = [self.solutions[idx_sympy_1], idx_sympy_1]
            sol_sympy_2 = [self.solutions[idx_sympy_2], idx_sympy_2]

            integrand = sp.Abs(sol_sympy_1[0].rhs - sol_sympy_2[0].rhs) ** ell_norm
            print(integrand)
            integral = self.compute_integral(integrand)

            loss[0, i] = sol_sympy_1[1]
            loss[1, i] = sol_sympy_2[1]
            if integral < self.max_integral_value:
                loss[2, i] = integral.item()
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

    def compute_integral(self, sympy_eq):
        func, symbols = self.eq_dataset.sympy_to_torch(sympy_eq)
        grids = self.create_discrete_grids(symbols)
        print(grids[0])
        _arg = {sym: _grid for sym, _grid in zip(symbols, grids)}
        return torch.mean(func(**_arg))

    def create_discrete_grids(self, symbols):
        grid = torch.linspace(*self.grid_size, device=self.eq_dataset.device)
        grids = [grid for i in symbols]
        mesh = torch.meshgrid(grids)
        return mesh
