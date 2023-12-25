from __future__ import annotations

import numpy as np
import sympy as sp

sp_function_to_numpy_function = {
    sp.Mul: np.multiply,
    sp.Add: np.add,
    sp.Pow: np.power,
    sp.exp: np.exp,
    sp.log: np.log,
    sp.sin: np.sin,
    sp.cos: np.cos,
    sp.tan: np.tan,
    sp.asin: np.arcsin,
    sp.acos: np.arccos,
    sp.atan: np.arctan,
    sp.sqrt: np.sqrt,
    sp.Abs: np.abs,
    sp.sign: np.sign,
    sp.Eq: np.equal,
}
