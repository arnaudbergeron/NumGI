from __future__ import annotations

import itertools
import random

import sympy as sp


class SolutionGenerator:
    """Generates solutions for contrived questions to be used for training data.

    Might  make sense to generate the equations in parallel here.
    """

    def __init__(
        self,
    ):
        self.PROB_NEW_SYMBOL = 0.3
        self.PROB_USED_SYMBOL = 1 - self.PROB_NEW_SYMBOL
        self.NEW_VARS = self.VARIABLES
        self.USED_VARS = []

    def generate_solution_dataset(
        self, min_ops: int, max_ops: int, num_eqs: int, vars: list, funcs: list, ops: list
    ) -> list:
        """Call to generate dataset of equations."""
        return [
            self.generate_solution(i, vars, funcs, ops)
            for i, _ in itertools.product(range(min_ops, max_ops + 1), range(num_eqs))
        ]

    def generate_solution(self, num_ops: int, vars: list, funcs: list, ops: list):
        """Generate a list of solution equations with a specific number of operations."""
        used_vars = []
        new_vars = vars.copy()
        for i in range(num_ops):
            op = self.choose_operation(ops)
            if i == 0:
                var = self.choose_variable(new_vars, used_vars)
                f = self.choose_function(funcs)
                f1 = f(var)
            else:
                if op[1] == "arithmetic":
                    var = self.choose_variable(new_vars, used_vars)
                    f2 = self.choose_function(funcs)
                    f2 = f2(var)
                    f1 = self.arithmetic_handler(op[0], f1, f2)
                elif op[1] == "differential":
                    var = self.choose_used_variable(used_vars=used_vars)
                    f1 = sp.Derivative(f1, var)
                elif op[1] == "integration":
                    var = self.choose_variable(new_vars, used_vars)
                    f1 = sp.Integral(f1, var)
                elif op[1] == "exponent":
                    exp = random.randint(1, 10)
                    f1 = sp.Pow(f1, exp)
        return f1

    def arithmetic_handler(self, operation: str, f1, f2):
        if operation == "addition":
            return sp.Add(f1, f2)
        elif operation == "division":
            return sp.Mul(f1, sp.Pow(f2, -1))
        elif operation == "multiplication":
            return sp.Mul(f1, f2)
        elif operation == "subtraction":
            f2 = sp.Mul(f2, -1)
            return sp.Add(f1, f2)

    def choose_function(self, functions: list | None):
        """Chooses a function from a list and returns a sympy function object."""
        return (
            random.choice(functions)
            if functions is not None
            else random.choice(self.DIFFERENTIAL_FUNCTIONS)
        )

    def choose_variable(self, new_vars: list | None, used_vars: list | None):
        """Chooses variables, calls determines which list should be used.

        Probabilities for when to choose which need to be initialized somewhere.
        """
        if used_vars is None or len(used_vars) <= 0 or random.random() < self.PROB_NEW_SYMBOL:
            var = self.pop_random(new_vars)
            used_vars.append(var)
            return sp.symbols(var)
        return sp.symbols(random.choice(used_vars))

    def choose_used_variable(self, used_vars: list):
        return sp.symbols(random.choice(used_vars))

    def choose_operation(
        self,
        operations: list | None,
    ):
        """From list of operations returns a sympy operation object and identifier.

        TODO: Add probabilities for when to choose which operation.
        """
        return (
            random.choice(operations) if operations is not None else random.choice(self.OPERATIONS)
        )

    def pop_random(self, lst: list):
        idx = random.randrange(len(lst))
        return lst.pop(idx)

    DIFFERENTIAL_FUNCTIONS = [
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

    OPERATIONS = [
        ("multiplication", "arithmetic"),
        ("addition", "arithmetic"),
        ("subtraction", "arithmetic"),
        ("division", "arithmetic"),
        ("differential", "differential"),
        ("integration", "integration"),
        ("exponent", "exponent"),
    ]
    VARIABLES = ["x", "y", "z", "beta", "gamma"]


if __name__ == "__main__":
    sg = SolutionGenerator()
    eqs = sg.generate_solution_dataset(
        min_ops=2,
        max_ops=7,
        num_eqs=4,
        vars=sg.VARIABLES,
        funcs=sg.DIFFERENTIAL_FUNCTIONS,
        ops=sg.OPERATIONS,
    )
    print(eqs)
