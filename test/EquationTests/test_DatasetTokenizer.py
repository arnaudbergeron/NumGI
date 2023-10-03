from __future__ import annotations

import sympy as sp

from NumGI.DatasetTokenizer import DatasetTokenizer


def test_DatasetTokenizer():
    sp_x = sp.symbols("x")
    sp_f = sp.Function("f")(sp_x)
    eq = sp.Eq(sp_f, sp_x)
    sol = sp.Eq(sp_f, sp_x)

    tokenizer = DatasetTokenizer([eq, eq], [sol, sol], True, True)

    eq = tokenizer.x_tokenized[0].tolist()
    sol = tokenizer.y_tokenized[0].tolist()

    eq = tokenizer.tokens_to_sympy(eq)
    sol = tokenizer.tokens_to_sympy(sol)

    assert (eq.replace(sol.lhs, sol.rhs)).doit()

    assert tokenizer is not None
    assert tokenizer.x_tokenized is not None
    assert tokenizer.y_tokenized is not None
    assert tokenizer.max_length_x is not None
    assert tokenizer.max_length_y is not None

    tokenizer.split(0.5)

    assert tokenizer.x_train is not None
    assert tokenizer.y_train is not None

    assert tokenizer.x_val is not None
    assert tokenizer.y_val is not None
