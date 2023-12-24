from __future__ import annotations

import numpy as np
import sympy as sp
import torch

from NumGI.ConstantDictionaries import DIFFERENTIAL_FUNCTIONS
from NumGI.ConstantDictionaries import OPERATIONS
from NumGI.EquationTokenizer import EquationTokenizer
from NumGI.SolutionGenerator import SolutionGenerator

sg = SolutionGenerator()
sg.PROB_NEW_SYMBOL = 0
n_eqs = 30
sols = [
    sg.generate_solution(4, ["x"], DIFFERENTIAL_FUNCTIONS, OPERATIONS)[0].simplify()
    for i in range(n_eqs)
]

tokenizer = EquationTokenizer()

test_arr = [1, 2, 5, 10, 20]
np_test = np.array(test_arr)
torch_test = torch.tensor(test_arr, device=tokenizer.device)
x = sp.Symbol("x")

cnt = 0

for i in sols:
    try:
        np_func, var = tokenizer.sympy_to_numpy(i)
        np_res = np_func(np_test).tolist()
    except TypeError:
        cnt += 1
        continue

    if cnt > n_eqs / 2:
        raise Exception(
            "Too many equations with TypeError are equations correctly generated \
                or error in sp to np func"
        )

    sp_res = []
    for idx, j in enumerate(test_arr):
        try:
            sp_res.append(float(i.replace(x, j).evalf()))
        except Exception as e:
            print(e)
            sp_res.append(np_res[idx])

    torch_func, var = tokenizer.sympy_to_torch(i)
    torch_res = torch_func(**{_arg: torch_test for _arg in var}).tolist()

    tol = 1e-3
    for i in range(len(sp_res)):
        try:
            if sp_res[i] is not None and np_res[i] is not None and torch_res[i] is not None:
                continue
            elif sp_res[i] == 0:
                assert (sp_res[i] - np_res[i]) < tol
                assert (sp_res[i] - torch_res[i]) < tol
            else:
                assert (sp_res[i] - np_res[i]) / sp_res[i] < tol
                assert (sp_res[i] - torch_res[i]) / sp_res[i] < tol
        except Exception as e:
            print(
                f"eq:{i}, sp_res: {sp_res[i]}, np_res: {np_res[i]}, torch_res: {torch_res[i]}, {e}"
            )
            raise
