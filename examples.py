from simplex_solver import SimplexSolver
from interior_point_solver import InteriorPointSolver


def main() -> None:
    print("Example 1")
    print("Interior point alpha = 0,5: ")
    InteriorPointSolver(
        mode=InteriorPointSolver.Mode.MAXIMIZE,
        c=[9, 10, 16, 0, 0, 0],
        a=[
            [18, 15, 12, 1, 0, 0],
            [6, 4, 8, 0, 1, 0],
            [5, 3, 3, 0, 0, 1]
        ],
        b=[360, 192, 180],
        start=[1, 1, 1, 315, 174, 169],
        alpha=0.5,
        eps=5
    ).solve()

    print("Interior point alpha = 0,9: ")
    InteriorPointSolver(
        mode=InteriorPointSolver.Mode.MAXIMIZE,
        c=[9, 10, 16, 0, 0, 0],
        a=[
            [18, 15, 12, 1, 0, 0],
            [6, 4, 8, 0, 1, 0],
            [5, 3, 3, 0, 0, 1]
        ],
        b=[360, 192, 180],
        start=[1, 1, 1, 315, 174, 169],
        alpha=0.9,
        eps=5
    ).solve()

    print()
    print("Simplex: ")
    SimplexSolver(
        mode=SimplexSolver.Mode.MAXIMIZE,
        c=[9, 10, 16],
        a=[
            [18, 15, 12],
            [6, 4, 8],
            [5, 3, 3]
        ],
        b=[360, 192, 180],
        eps=5
    ).solve()

    print()
    print("--> Example 2. An unbounded objective function")

    print("Interior point alpha = 0,5: ")
    InteriorPointSolver(
        mode=InteriorPointSolver.Mode.MAXIMIZE,
        c=[5, 4, 0, 0],
        a=[
            [1, 0, 1, 0],
            [1, -1, 0, 1]
        ],
        b=[7, 8],
        start=[1, 1, 6, 8],
        alpha=0.5,
        eps=5
    ).solve()

    print("Interior point alpha = 0,9: ")
    InteriorPointSolver(
        mode=InteriorPointSolver.Mode.MAXIMIZE,
        c=[5, 4, 0, 0],
        a=[
            [1, 0, 1, 0],
            [1, -1, 0, 1]
        ],
        b=[7, 8],
        start=[1, 1, 6, 8],
        alpha=0.9,
        eps=5
    ).solve()

    print()
    print("Simplex: ")
    SimplexSolver(
        mode=SimplexSolver.Mode.MAXIMIZE,
        c=[5, 4],
        a=[
            [1, 0],
            [1, -1]
        ],
        b=[7, 8],
        eps=5
    ).solve()

    print("--> Example 6 - More variables. Taken from lab 2, task 1")

    print("Interior point alpha = 0,5: ")
    InteriorPointSolver(
        mode=InteriorPointSolver.Mode.MAXIMIZE,
        c=[100, 140, 120, 0, 0, 0, 0],
        a=[
            [3, 6, 7, 1, 0, 0, 0],
            [2, 1, 8, 0, 1, 0, 0],
            [1, 1, 1, 0, 0, 1, 0],
            [5, 3, 3, 0, 0, 0, 1]
        ],
        b=[135, 260, 220, 360],
        start=[1, 1, 1, 119, 249, 117, 349],
        alpha=0.5,
        eps=5

    ).solve()

    print("Intetior point alpha = 0,9: ")
    InteriorPointSolver(
        mode=InteriorPointSolver.Mode.MAXIMIZE,
        c=[100, 140, 120, 0, 0, 0, 0],
        a=[
            [3, 6, 7, 1, 0, 0, 0],
            [2, 1, 8, 0, 1, 0, 0],
            [1, 1, 1, 0, 0, 1, 0],
            [5, 3, 3, 0, 0, 0, 1]
        ],
        b=[135, 260, 220, 360],
        start=[1, 1, 1, 119, 249, 117, 349],
        alpha=0.9,
        eps=5
    ).solve()

    print("Simplex: ")
    SimplexSolver(
        mode=SimplexSolver.Mode.MAXIMIZE,
        c=[100, 140, 120],
        a=[
            [3, 6, 7],
            [2, 1, 8],
            [1, 1, 1],
            [5, 3, 3]
        ],
        b=[135, 260, 220, 360],
        eps=5,
    ).solve()

    print("--> Example 7 - Minimization, taken from tutorial")
    print("Interior point alpha = 0,5: ")
    solver = InteriorPointSolver(
        mode=InteriorPointSolver.Mode.MINIMIZE,
        c=[-2, 2, -6, 0, 0, 0],
        a=[
            [2, 1, -2, 1, 0, 0],
            [1, 2, 4, 0, 1, 0],
            [1, -1, 2, 0, 0, 1]
        ],
        b=[24, 23, 10],
        start=[1, 1, 1, 23, 16, 8],
        alpha=0.5,
        eps=5
    ).solve()

    print("Intetior point alpha = 0,9: ")
    InteriorPointSolver(
        mode=InteriorPointSolver.Mode.MINIMIZE,
        c=[-2, 2, -6, 0, 0, 0],
        a=[
            [2, 1, -2, 1, 0, 0],
            [1, 2, 4, 0, 1, 0],
            [1, -1, 2, 0, 0, 1]
        ],
        b=[24, 23, 10],
        start=[1, 1, 1, 23, 16, 8],
        alpha=0.9,
        eps=5
    ).solve()

    print("Simplex: ")
    SimplexSolver(
        mode=InteriorPointSolver.Mode.MINIMIZE,
        c=[-2, 2, -6],
        a=[
            [2, 1, -2],
            [1, 2, 4],
            [1, -1, 2]
        ],
        b=[24, 23, 10],
        eps=5
    ).solve()


if __name__ == "__main__":
    main()
