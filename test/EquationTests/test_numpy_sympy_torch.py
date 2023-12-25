from __future__ import annotations

import math

import numpy as np
import sympy as sp
import torch

from NumGI.ConstantDictionaries import DIFFERENTIAL_FUNCTIONS
from NumGI.EquationTokenizer import EquationTokenizer
from NumGI.SolutionGenerator import SolutionGenerator


def test_sp_np_torch():
    sg = SolutionGenerator()
    sg.PROB_NEW_SYMBOL = 0
    n_eqs = 30
    sols = [
        # sg.generate_solution(4, ["x"], DIFFERENTIAL_FUNCTIONS, OPERATIONS)[0].simplify()
        # for i in range(n_eqs)
    ]

    for func in DIFFERENTIAL_FUNCTIONS:
        sols.append(func(sp.Symbol("x")))

    tokenizer = EquationTokenizer()

    test_arr = [-10, -5, -2, -1, 0, 1, 2, 5, 10, 20]
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
            print("typeerr")
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

        tol = 1e-4
        for idx in range(len(sp_res)):
            print(
                f"eq:{i}, sp_res: {sp_res[idx]}, np_res: {np_res[idx]}, torch_res: {torch_res[idx]}"
            )
            try:
                if math.isnan(sp_res[idx]) or math.isnan(np_res[idx]) or math.isnan(torch_res[idx]):
                    continue
                elif sp_res[idx] == 0:
                    assert (sp_res[idx] - np_res[idx]) < tol
                    assert (sp_res[idx] - torch_res[idx]) < tol
                elif math.isinf(np_res[idx]):
                    assert np_res[idx] == sp_res[idx]
                    assert np_res[idx] == torch_res[idx]
                else:
                    assert (sp_res[idx] - np_res[idx]) / sp_res[idx] < tol
                    assert (sp_res[idx] - torch_res[idx]) / sp_res[idx] < tol
            except Exception as e:
                print(
                    f"eq:{i}, sp_res: {sp_res[idx]}, np_res: {np_res[idx]}, \
                        torch_res: {torch_res[idx]}, {e}"
                )
                raise


if __name__ == "__main__":
    test_sp_np_torch()
