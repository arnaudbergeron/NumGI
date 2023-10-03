from __future__ import annotations

import multiprocessing as mp
import os

import sympy as sp
import torch

from NumGI.DatasetTokenizer import DatasetTokenizer
from NumGI.EquationTokenizer import EquationTokenizer
from NumGI.SolutionGenerator import SolutionGenerator


def worker(args):
    sols = args[0].generate_solution_dataset(*args[1:-1])
    generate_tokenized_lists(sols, args[-1])
    return [args[-1]]


def generate_tokenized_lists(sols, num):
    """Generates tokenized lists of equations and solutions. Saves them to disk."""
    x = []
    y = []
    for i in sols:
        if not isinstance(i[1], sp.logic.boolalg.BooleanTrue) and not isinstance(
            i[1], sp.logic.boolalg.BooleanFalse
        ):
            x.append(i[0])
            y.append(i[1])

    tok = EquationTokenizer()
    y_list = [tok.sympy_to_list(i) for i in y]
    x_nozoo = []
    y_nozoo = []
    for idx, i in enumerate(y_list):
        if len(i) < 5_000:
            if "zoo" not in [str(j) for j in i]:
                x_nozoo.append(x[idx])
                y_nozoo.append(y[idx])

    dataset = DatasetTokenizer(x_nozoo, y_nozoo, useDefaultTokenizer=True)

    torch.save(dataset.x_tokenized, f"data/new_sol_generator/x_{num}.pt")
    torch.save(dataset.y_tokenized, f"data/new_sol_generator/y_{num}.pt")


def generate_eq_parallel(gen_args: list, path: str, num_thousands: int):
    """Generates equations in parallel.

    Note some equations will be discarded because they are too long.
    This won't create the exact number of expected equations.

    Args:
        path (str): path to save the equations to
        num_thousands (int): number of thousands of equations to generate
    """
    pool = mp.Pool(mp.cpu_count() - 1)
    shift = 0
    solgen = SolutionGenerator()

    for i in os.listdir(path):
        new_i = (i.split("_")[1]).split(".")[0]
        shift = max(int(new_i), shift)

    shift += 1
    # Define the parameters for each call to generate_solution_dataset
    parameters = [([solgen] + gen_args + [shift + _]) for _ in range(num_thousands)]

    pool.map(worker, parameters)


if __name__ == "__main__":
    diff_func = [
        sp.sin,
        sp.cos,
        sp.tan,
        sp.cot,
        sp.sec,
        sp.csc,
        sp.exp,
        sp.log,
        sp.sqrt,
        sp.asin,
        sp.acos,
        sp.atan,
        sp.acot,
        sp.asec,
        sp.acsc,
        sp.sinh,
        sp.cosh,
        sp.tanh,
        sp.coth,
        sp.sech,
        sp.csch,
        sp.asinh,
        sp.acosh,
        sp.atanh,
        sp.acoth,
        sp.asech,
        sp.acsch,
    ]
    ops = [
        ("multiplication", "arithmetic"),
        ("addition", "arithmetic"),
        ("subtraction", "arithmetic"),
        ("division", "arithmetic"),
        ("differential", "differential"),
        ("exponent", "exponent"),
    ]
    vars = ["x", "y", "z", "beta", "gamma", "delta", "a", "b", "c", "d", "epsilon"]
    gen_args = [
        (3, 10),
        (3, 5),
        1_00,
        vars,
        diff_func,
        ops,
    ]
    generate_eq_parallel(gen_args, "data/new_sol_generator", 5)
