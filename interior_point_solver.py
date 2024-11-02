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
        Perform one iteration of the Interior point method
        :return: whether to continue iterating. [False] if this was the last step
        """

        # Get the index of the pivot column
        x = self.start
        self.c = np.transpose(self.c)
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
                break

            if np.linalg.det(f) == 0:
                x = None
                self.is_not_applicable = True
                break

            f_inverse = np.linalg.inv(f)
            H = np.dot(np.transpose(aa), f_inverse)
            P = np.subtract(I, np.dot(H, aa))
            cp = np.dot(P, cc)

            if np.any(np.isnan(cp)) or np.any(np.isinf(cp)):
                x = None
                break

            if np.all(cp >= 0):
                x = None
                break  # Function is unbounded, leave method flow

            nu = np.absolute(np.min(cp))
            if nu < 1e-10:
                self.is_not_applicable = True
                x = None
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
        or print "The method is not applicable!" or "The problem has no solution!" in those special cases.
        :return: a tuple (solution, X*) or [None] in special cases
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
