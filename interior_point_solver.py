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

    def __init__(self, mode: Mode, start: list[float], c: list[float], a: list[list[float]], b: list[float], alpha: float, eps: int) -> None:
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
            y = np.add(np.ones(len(self.c), float), (self.alpha/nu) * cp)
            yy = np.dot(d, y)
            x = yy
            if norm(np.subtract(yy, v), ord = 2) <  0.00001:
                break
            i+= 1
        return x

    def solve(self) -> Union[tuple[float, list[float]], None]:
        """
        Solve the problem in this solver and print the solution and X*,
        or "Unbounded" if the objective function is unbounded
        :return: a tuple (solution, X*) or [None] if the objective function is unbounded
        """
        x =self.calculate()




        if not self.is_not_applicable and x is not None:
            x = [round(float(i),self.eps) for i in x]
            for i in range(len(x)):
                self.solution += self.c[i]*x[i]

            if self.mode == InteriorPointSolver.Mode.MINIMIZE:  # Flip the solution in case we were minimizing
                self.solution *= -1
            print("X: ", x)
            print("Solution:",self.solution)
            return self.solution, x
        elif self.is_not_applicable:
            print("The method is not applicable!")
            return None
        else:
            print("The problem does not have solution!")
            return None


def interior_point_solve_and_check(
        mode: InteriorPointSolver.Mode,
        start: list[float],
        objective_function: list[float],
        constraints_matrix: list[list[float]],
        constraints_right_hand_side: list[float],
        alpha: float,
        epsilon: int,
        expected_solution: Union[float, None],
) -> None:
    """
    Run the simplex solver with given data and assert-check the solution.

    :param mode: [InteriorPointSolver] mode on whether to minimize or maximize the function
    :param objective_function: coefficients of the objective function F(x_1, ..., x_m) = 0
    :param constraints_matrix: coefficients of constraints equations Q_i (x_1, ..., x_m) <= rhs_i
    :param constraints_right_hand_side: results of the constraints equations
    :param epsilon: optimization accuracy. Number of digits after the floating point to consider
    :param expected_solution: the expected solution of the optimization problem
    """
    solver = InteriorPointSolver(mode,start, objective_function, constraints_matrix, constraints_right_hand_side,alpha, epsilon)
    solutions,x = solver.solve()

    if solutions is None:  # Unbounded function
        if expected_solution is not None:
            raise ArithmeticError("Expected an unbounded function")
    else:
        if solutions[0] != expected_solution:
            raise ArithmeticError(f"Expected solution {expected_solution}, got {solutions[0]}")

        answer = 0
        for i in range(len(objective_function)):
            answer += objective_function[i] * solutions[1][i]

        if not isclose(answer, expected_solution, rel_tol=10**(-epsilon)):
            raise ArithmeticError("X* coefficients do not produce expected solution")

        for i in range(len(constraints_right_hand_side)):
            calculated = sum([constraints_matrix[i][j] * solutions[1][j] for j in range(len(solutions[1]))])
            if calculated > constraints_right_hand_side[i]:
                raise ArithmeticError(f"Produced X* does not match {i}-th constraint")

def read_data():
    _, ln_start, _, ln_C, _, *lns_A, _, ln_b = (ln.strip() for ln in stdin.readlines())
    start = [float(x) for x in ln_start.strip().split()]
    C = [float(x) for x in ln_C.strip().split()]
    A = [[float(x) for x in ln.strip().split()] for ln in lns_A]
    b = [float(x) for x in ln_b.strip().split()]
    return C, A, b, start


print("Example 1")
print("Intetior point alpha = 0,5")
solver = InteriorPointSolver(mode=InteriorPointSolver.Mode.MINIMIZE,
                             c=[-2, 2, -6, 0, 0, 0],
                             a=[
                                 [2, 1, -2, 1, 0, 0],
                                 [1, 2, 4, 0, 1, 0],
                                 [1, -1, 2, 0, 0, 1]
                             ],
                             b=[24, 23, 10],
                             start=[1, 1, 1, 23, 16, 8],
                             alpha=0.5,
                             eps=5)

solver.solve()
# if sol is not None:
#     print("ans for alpha = 0.5")
#     print("X: ", x1)
#     print(sol)
# else:
#     print("ans for alpha = 0.5")


print("Intetior point alpha = 0,9")
solver = InteriorPointSolver(mode=InteriorPointSolver.Mode.MINIMIZE,
                             c=[-2, 2, -6, 0, 0, 0],
                             a=[
                                 [2, 1, -2, 1, 0, 0],
                                 [1, 2, 4, 0, 1, 0],
                                 [1, -1, 2, 0, 0, 1]
                             ],
                             b=[24, 23, 10],
                             start=[1, 1, 1, 23, 16, 8],
                             alpha=0.9,
                             eps=5)

solver.solve()


print("Simplex")
solver = SimplexSolver(mode=InteriorPointSolver.Mode.MINIMIZE,
                       c=[-2, 2, -6],
                       a=[
                           [2, 1, -2],
                           [1, 2, 4],
                           [1, -1, 2]
                       ],
                       b=[24, 23, 10],
                       eps=5)

sol,x1 =solver.solve()