from __future__ import annotations

import sympy as sp

from NumGI.DatasetTokenizer import DatasetTokenizer


def test_DatasetTokenizer():
    sp_x = sp.symbols("x")
    eq = sp.Eq(sp.sin(sp_x), sp.symbols("y"))
    sol = sp.Eq(sp.Function("f")(sp_x), 1)

    tokenizer = DatasetTokenizer([eq, eq], [sol, sol])

    assert tokenizer is not None
    assert tokenizer.x_tokenized is not None
    assert tokenizer.y_tokenized is not None
    assert tokenizer.max_length is not None

    tokenizer.split(0.5)

    assert tokenizer.x_train is not None
    assert tokenizer.y_train is not None

    assert tokenizer.x_val is not None
    assert tokenizer.y_val is not None
