from __future__ import annotations

import random


class SolutionGenerator:
    """Generates solutions for contrived questions to be used for training data.

    Might  make sense to generate the equations in parallel here.
    """

    def generate_solution_dataset(
        self, min_ops: int, max_ops: int, num_eqs: int, vars: list, funcs: list, ops: list
    ) -> list:
        """Call to generate dataset of equations."""
        dataset = []
        for i in range(min_ops, max_ops):
            for _ in range(num_eqs):
                dataset.append(self.generate_solution(i, vars, funcs, ops))
        return dataset

    def generate_solution(self, num_ops: int, vars: list, funcs: list, ops: list):
        """Generate a list of solution equations with a specific number of operations."""
        # used_vars = [], new_vars = vars
        for i in range(num_ops):
            op, op_type = self.choose_operation(ops)
            if i == 0:
                if op_type == "arithmetic":
                    f1 = self.arithmetic_intializer(op)
                elif op_type == "differential":
                    f1 = self.differential_initializer(op)
                elif op_type == "integration":
                    f1 = self.integration_initializer(op)
            else:
                if op_type == "arithmetic":
                    print()
                    # var = self.choose_variable(new_vars, used_vars)
                    # f2 = self.choose_function(funcs)
                    # f1 = op(f1,f2) how to handle this needs to be implemented
                elif op_type == "differential":
                    print()
                    # var = self.choose_variable(used_vars=used_vars)
                    # f1 = sp.derivative(f1, var)
                elif op_type == "integration":
                    print()
                    # var = self.choose_variable(new_vars, used_vars)
                    # f1 = integrate(f1, var) might need to add bounds but for later
        return f1

    def choose_function(self, functions: list):
        """Chooses a function from a list and returns a sympy function object."""

    def choose_variable(self, new_vars: list | None, used_vars: list | None):
        """Chooses variables, calls determines which list should be used.

        Probabilities for when to choose which need to be initialized somewhere.
        """

    def choose_variable_helper(self, vars: list):
        """Return a random selection from vars.

        Might need to be changed depending on how the vars list works. Might make
        sense to store the actual sympy objects in a dictionary and use the vars
        list to choose the key.
        """
        return random.choice(vars)

    def integration_initializer(self, op: str):
        """Initialize function if operation is integration."""

    def differential_initializer(self, op: str):
        """Initialize function if operation is differential."""

    def arithmetic_intializer(self, op: str):
        """Initialize function if operation is arithmetic."""

    def choose_operation(
        self,
        operations: list,
    ):
        """From list of operations returns a sympy operation object and identifier."""
        return "add", "arithmetic"
