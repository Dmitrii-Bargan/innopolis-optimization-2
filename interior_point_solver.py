from enum import Enum
from sys import stdin
from typing import Union
from math import isclose
from simplex_solver import SimplexSolver

import numpy as np
from numpy.linalg import norm


class InteriorPointSolver:
    class Mode(str, Enum):
        MAXIMIZE = "Maximize"
        MINIMIZE = "Minimize"

    def __init__(
            self,
            mode: Mode,
            start: list[float],
            c: list[float],
            a: list[list[float]],
            b: list[float],
            alpha: float,
            eps: int
    ) -> None:
        """
        Construct Interior-Point Algorithm problem solver

        :param mode: one of [Mode.MAXIMIZE] or [Mode.MINIMIZE]
        :param c: coefficients of the objective function
        :param a: matrix of the coefficients of the constraints
        :param b: right hand side of the constraints equations
        :param eps: solution accuracy. How many digits after the floating point to consider
        """

        self.mode: InteriorPointSolver.Mode = mode
        """Mode of this problem solver"""

        self.actual_coefficients: list[float] = c
        """Coefficients of the objective function regardless of whether we are maximizing or minimizing"""

        self.c: list[float] = c if mode == InteriorPointSolver.Mode.MAXIMIZE else [-j for j in c]
        """Coefficients of the objective function. Inverted if we are minimizing"""

        self.a: list[list[float]] = a
        """Matrix of the coefficients of the constraints"""

        self.b: list[float] = b
        """Right hand side of the constraints equations"""

        self.start: list[float] = start
        """Initial point"""

        self.alpha: float = alpha
        """alpha in Interior-Point Algorithm"""

        self.eps: int = eps
        """Solution accuracy"""

        self.solution = 0
        """Optimal solution for this problem"""

        self.is_not_applicable = False
        """Whether the objective function is unbounded"""

    def calculate(self):
        """
        Perform one iteration of the Simplex method
        :return: whether to continue iterating. [False] if this was the last step
        """

        # Get the index of the pivot column
        x = self.start
        self.c = np.transpose(self.c)
        exception = None
        i = 1

        while True:
            v = x
            d = np.diag(x)
            aa = np.dot(self.a, d)
            cc = np.dot(d, self.c)
            I = np.eye(len(self.c))
            f = np.dot(aa, np.transpose(aa))

            if np.any(np.isnan(f)) or np.any(np.isinf(f)):
                x = None
                exception = "The problem does not have solution!"
                break

            if np.linalg.det(f) == 0:
                x = None
                self.is_not_applicable = True
                exception = "The method is not applicable!"
                break

            f_inverse = np.linalg.inv(f)
            H = np.dot(np.transpose(aa), f_inverse)
            P = np.subtract(I, np.dot(H, aa))
            cp = np.dot(P, cc)

            if np.any(np.isnan(cp)) or np.any(np.isinf(cp)):
                x = None
                exception = "The problem does not have solution!"
                break

            if np.all(cp >= 0):
                x = None
                exception = "The problem does not have solution!"
                break  # Решение неограниченно, выход из функции

            nu = np.absolute(np.min(cp))
            if nu < 1e-10:
                x = None
                exception = "The problem does not have solution!"
                break

            y = np.add(np.ones(len(self.c), float), (self.alpha / nu) * cp)
            yy = np.dot(d, y)
            x = yy

            if norm(np.subtract(yy, v), ord=2) < 0.00001:
                break

            i += 1
        return x

    def solve(self) -> Union[tuple[float, list[float]], None]:
        """
        Solve the problem in this solver and print the solution and X*,
        or "Unbounded" if the objective function is unbounded
        :return: a tuple (solution, X*) or [None] if the objective function is unbounded
        """
        x = self.calculate()

        if not self.is_not_applicable and x is not None:
            x = [round(float(i), self.eps) for i in x]
            for i in range(len(x)):
                self.solution += self.c[i] * x[i]

            if self.mode == InteriorPointSolver.Mode.MINIMIZE:  # Flip the solution in case we were minimizing
                self.solution *= -1

            print("X: ", x)
            print("Solution:", self.solution)
            return self.solution, x
        elif self.is_not_applicable:
            print("The method is not applicable!")
            return None
        else:
            print("The problem does not have solution!")
            return None


def main() -> None:
    while True:
        mode = input('Maximization or minimization? [max|min|maximization|minimization|maximize|minimize]: ')
        if mode in ['max', 'maximization', 'maximize']:
            mode = InteriorPointSolver.Mode.MAXIMIZE
            break
        elif mode in ['min', 'minimization', 'minimize']:
            mode = InteriorPointSolver.Mode.MINIMIZE
            break
        else:
            print('Invalid input!')

    c = []
    c_size = int(input('Size of the vector C: '))
    print('Enter components of vector C (one on each line):')
    for _ in range(c_size):
        c.append(float(input()))

    a = []
    print('Constructing the matrix A:')
    num_constraints = int(input('Number of constraints: '))
    for i in range(num_constraints):
        array = [float(x) for x in input(f'Enter values for constraint {i + 1} (space-separated): ').strip().split(' ')]
        a.append(array)

    b = []
    print('Enter components of vector b:')
    for _ in range(num_constraints):
        b.append(float(input()))

    x_size = int(input('Size of the vector x_0: '))
    x_0 = []
    print('Enter components of x_0: ')
    for _ in range(x_size):
        x_0.append(float(input()))

    eps = int(input('Enter solution accuracy: '))

    print('--> Solving with alpha 0.5')
    InteriorPointSolver(mode, x_0, c, a, b, 0.5, eps).solve()

    print('--> Solving with alpha 0.9')
    InteriorPointSolver(mode, x_0, c, a, b, 0.9, eps).solve()


if __name__ == '__main__':
    main()
