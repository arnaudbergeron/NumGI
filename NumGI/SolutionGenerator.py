from __future__ import annotations

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
        self, ops_sol: tuple, ops_eq: tuple, num_eqs: int, vars: list, funcs: list, ops: list
    ) -> list:
        """Call to generate dataset of equations."""
        dataset = []
        for _ in range(num_eqs):
            for i in range(ops_sol[0], ops_sol[1]):
                sol, used_vars = self.generate_solution(i, vars, funcs, ops)
                equation = self.generate_equation(used_vars, ops_eq, ops, sol)
                dataset.append((sol, equation))

    def generate_equation(self, used_vars, ops_eqs, ops, sol):
        """Generate an equation from a solution."""
        tree = self.generate_equation_tree(random.randint(ops_eqs[0], ops_eqs[1]), ops)
        return self.tree_to_equation(tree, sol, used_vars)

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
        return f1, used_vars

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

    def tree_to_equation(
        self,
        tree: EquationTree,
        sol,
        used_vars: list,
    ):
        """Converts a tree to a sympy equation."""
        root = tree.root

        vars = [sp.Symbol(var) for var in used_vars]
        func = sp.Function("f")(*vars)
        expression = self.tree_to_eq_helper(root, sol, used_vars)
        equation = sp.Eq(expression, expression.doit())
        equation.replace(expression, func)
        return equation

    def tree_to_eq_helper(self, node: EquationTree.Node, sol, used_vars: list):
        if node.op[1] == "arithmetic":
            return self.arithmetic_handler(
                node.op[0],
                self.tree_to_eq_helper(node.left, sol, used_vars),
                self.tree_to_eq_helper(node.right, sol, used_vars),
            )
        elif node.op[1] == "differential":
            return sp.Derivative(
                self.tree_to_eq_helper(node.left, sol, used_vars),
                self.tree_to_eq_helper(node.right, sol, used_vars),
            )
        elif node.op[1] == "integration":
            return sp.Integral(
                self.tree_to_eq_helper(node.left, sol, used_vars),
                self.tree_to_eq_helper(node.right, sol, used_vars),
            )
        elif node.op[1] == "exponent":
            return sp.Pow(
                self.tree_to_eq_helper(node.left, sol, used_vars),
                self.tree_to_eq_helper(node.right, sol, used_vars),
            )
        elif node.op[1] == "symbol":
            return self.choose_used_variable(used_vars)
        elif node.op[1] == "number":
            return random.randint(1, 5)
        elif node.op[1] == "undefined" or node.op[1] == "function":
            return sol

    def generate_equation_tree(self, num_ops: int, ops: list):
        tree = self.EquationTree(self.EquationTree.Node(("function", "function"), None, None, 0))
        for i in range(num_ops):
            if i >= num_ops - 1:
                op = self.choose_op_noarithmetic(ops)
            else:
                op = self.choose_operation(ops)
            level = random.randint(0, tree.level)
            old_node = random.choice(tree.get_nodes_at_level(level))
            new_node = self.EquationTree.Node(op, None, None, level)
            tree.insert(old_node, new_node)
        return tree

    class EquationTree:
        """Tree structure for equations."""

        def __init__(self, root: Node):
            self.level = 0
            self.root = root

        def get_nodes_at_level(self, level: int):
            root = self.root
            q = [(root, root.level)]

            nodes_at_level = []
            while q:
                node, lvl = q.pop(0)
            if lvl == level:
                # TODO: add dummy values so that symbols can be recognized
                nodes_at_level.append(node) if node.op[1] != "symbol" or node.op[
                    1
                ] != "number" else None
            elif node.left:
                q.append((node.left, node.left.level))
            elif node.right:
                q.append((node.right, node.right.level))

            return nodes_at_level

        def insert(self, old_node: Node, new_node: Node):
            old = old_node
            old_node = new_node
            old_node.right = old
            self.update_level(old_node.right)
            if new_node.op[1] == "differential":
                old_node.left = self.Node(("symbol", "symbol"), None, None, old_node.level + 1)
            elif new_node.op[1] == "integration":
                old_node.left = self.Node(("symbol", "symbol"), None, None, old_node.level + 1)
            elif new_node.op[1] == "exponent":
                old_node.left = self.Node(("number", "number"), None, None, old_node.level + 1)
            elif new_node.op[1] == "arithmetic":
                old_node.left = self.Node(
                    ("operation", "undefined"), None, None, old_node.level + 1
                )

        def update_level(self, node: Node):
            node.level += 1
            self.level = max(self.level, node.level)
            if node.left:
                self.update_level(node.left)
            if node.right:
                self.update_level(node.right)

        class Node:
            """Node for equation tree."""

            def __init__(
                self,
                op: tuple,
                left,
                right,
                level: int,
            ):
                self.op = op
                self.left = left
                self.right = right
                self.level = level

    def choose_op_noarithmetic(self, ops: list):
        return random.choice(ops[3:])

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
    # must not be rordered need first three to be arithmetic because
    #  I am lazy and chooseop_noarithmetic is not implemented well
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


6
if __name__ == "__main__":
    sg = SolutionGenerator()
    eqs = sg.generate_solution_dataset(
        ops_sol=(3, 5),
        ops_eq=(2, 5),
        num_eqs=4,
        vars=sg.VARIABLES,
        funcs=sg.DIFFERENTIAL_FUNCTIONS,
        ops=sg.OPERATIONS,
    )
    print(eqs)
