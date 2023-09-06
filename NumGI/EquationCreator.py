import sympy as sp
import random
import numba


class EquationCreator:
    """
    This is the first version of the DE creator that aribirarily creates DEs.
    This was made purposefully simple as a test of the convergences of the first models.
    We want to find a good loss function, we will make this more general as time goes on.
    """
    def __init__(self, order=3):
        self.order = order

        # List of differential functions available in Sympy
        self.differential_functions = [
            sp.sin, sp.cos, sp.tan, sp.cot, sp.sec, sp.csc,
            sp.exp, sp.log, sp.sqrt, sp.asin, sp.acos, sp.atan,
            sp.acot, sp.asec, sp.acsc, sp.sinh, sp.cosh, sp.tanh,
            sp.coth, sp.sech, sp.csch, sp.asinh, sp.acosh, sp.atanh,
            sp.acoth, sp.asech, sp.acsch
        ]

    @numba.jit()
    def generate_random_function(self):
        #Generate a random function that will be the solution to the DE

        # Randomly select a differential function from the list
        differential_function = random.choice(DIFFERENTIAL_FUNCTIONS)

        # Generate a random variable for the function
        x = sp.symbols('x') #will need to make this more general

        # Apply the differential function to the variable
        function = differential_function(x)

        # Generate a random exponent for the function
        exponent = random.randint(1, 5)
        function = function**exponent

        # Generate a random coefficient for the function
        coefficient_p = random.randint(1, 10)
        coefficient_q = random.randint(1, 10)
        coefficient = sp.Rational(coefficient_p, coefficient_q)

        function = coefficient * function

        return function
    
    @numba.jit()
    def generate_random_differential_equation(self, function, num_op = 3):
        # Generate a random differential equation with solution: function

        # Generate a random order for the differential equation
        order = [random.randint(1, 3) for i in range(num_op)]

        # Generate a random differential operator based on the order
        x = sp.symbols('x')
        differential_operator = [sp.Derivative(function, x, i) for i in order]

        sum_or_mul = [random.randint(0, 1) for i in range(num_op-1)]
        lhs = differential_operator[0]
        for idx, num in enumerate(sum_or_mul):
            if num == 0:
                lhs = sp.Add(lhs, differential_operator[idx+1])
            else:
                lhs = sp.Mul(lhs,differential_operator[idx+1])

        #calculate the rhs
        rhs = lhs.doit()

        # Construct the differential equation
        differential_equation = sp.Eq(lhs, rhs)

        differential_equation = differential_equation.replace(function, sp.Function('f')(x))

        return differential_equation

    def __next__(self):
        func = self.generate_random_function()
        return func, self.generate_random_differential_equation(func, self.order)

    def __iter__(self):
        return self


DIFFERENTIAL_FUNCTIONS = [
            sp.sin, sp.cos, sp.tan, sp.cot, sp.sec, sp.csc,
            sp.exp, sp.log, sp.sqrt, sp.asin, sp.acos, sp.atan,
            sp.acot, sp.asec, sp.acsc, sp.sinh, sp.cosh, sp.tanh,
            sp.coth, sp.sech, sp.csch, sp.asinh, sp.acosh, sp.atanh,
            sp.acoth, sp.asech, sp.acsch
        ]