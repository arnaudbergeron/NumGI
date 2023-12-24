from __future__ import annotations

import sympy as sp
import torch

SP_TO_TORCH = {
    sp.sin: torch.sin,
    sp.cos: torch.cos,
    sp.tan: torch.tan,
    sp.csc: torch.csc,
    sp.exp: torch.exp,
    sp.log: torch.log,
    sp.sqrt: torch.sqrt,
    sp.asin: torch.asin,
    sp.acos: torch.acos,
    sp.atan: torch.atan,
    sp.acsc: torch.acsc,
    sp.sinh: torch.sinh,
    sp.cosh: torch.cosh,
    sp.tanh: torch.tanh,
    sp.csch: torch.csch,
    sp.asinh: torch.asinh,
    sp.acosh: torch.acosh,
    sp.atanh: torch.atanh,
    sp.acsch: torch.acsch,
}

DIFFERENTIAL_FUNCTIONS = [
    sp.sin,
    sp.cos,
    sp.tan,
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
    sp.csch,
    sp.asinh,
    sp.acosh,
    sp.atanh,
    sp.acsch,
]

OPERATIONS = [
    ("multiplication", "arithmetic"),
    ("addition", "arithmetic"),
    ("subtraction", "arithmetic"),
    ("division", "arithmetic"),
    ("differential", "differential"),
    # ("integration", "integration"),
    ("exponent", "exponent"),
]

VARIABLES = ["x", "y", "z", "beta", "gamma"]

DEFAULT_DICT = {
    ")": 0,
    sp.acsc: 1,
    sp.acot: 2,
    sp.asech: 3,
    sp.core.containers.Tuple: 4,
    "/": 5,
    sp.sech: 6,
    "END": 7,
    sp.exp: 8,
    "7": 9,
    "0": 10,
    sp.asin: 11,
    "5": 12,
    sp.core.function.Derivative: 13,
    "8": 14,
    sp.asec: 15,
    sp.core.add.Add: 16,
    sp.core.power.Pow: 17,
    sp.csch: 18,
    "START": 19,
    sp.csc: 20,
    "PAD": 21,
    sp.sin: 22,
    ",": 23,
    sp.acsch: 24,
    sp.core.relational.Equality: 25,
    "(": 26,
    "2": 27,
    sp.Symbol("x"): 28,
    sp.coth: 29,
    sp.Symbol("y"): 30,
    sp.log: 31,
    sp.cos: 32,
    "6": 33,
    sp.core.mul.Mul: 34,
    sp.acos: 35,
    "9": 36,
    sp.Function("f"): 37,
    "-": 38,
    sp.sqrt: 39,
    sp.cosh: 40,
    sp.tan: 41,
    sp.tanh: 42,
    sp.Symbol("z"): 43,
    "4": 44,
    "3": 45,
    sp.cot: 46,
    sp.asinh: 47,
    sp.atan: 48,
    sp.acosh: 49,
    "1": 50,
    sp.atanh: 51,
    ".": 52,
    sp.sinh: 53,
    sp.acoth: 54,
    sp.sec: 55,
    sp.Symbol("beta"): 56,
    sp.Symbol("gamma"): 57,
    sp.Symbol("delta"): 58,
    sp.Symbol("a"): 59,
    sp.Symbol("b"): 60,
    sp.Symbol("c"): 61,
    sp.Symbol("d"): 62,
    sp.Symbol("epsilon"): 63,
}
