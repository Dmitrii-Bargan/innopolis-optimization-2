from simplex_solver import SimplexSolver
from interior_point_solver import InteriorPointSolver


def main() -> None:
    print("Example 1")
    print()
    print("Intetior point alpha = 0,5: ")
    solver = InteriorPointSolver(mode=InteriorPointSolver.Mode.MAXIMIZE,
                                 c=[9, 10, 16, 0, 0, 0],
                                 a=[
                                     [18, 15, 12,1,0,0],
                                     [6, 4, 8,0,1,0],
                                     [5, 3, 3,0,0,1]
                                 ],
                                 b=[360, 192, 180],
                                 start=[1, 1, 1, 315, 174, 169],
                                 alpha=0.5,
                                 eps=5)

    solver.solve()

    print()
    print("Interior point alpha = 0,9: ")
    solver = InteriorPointSolver(mode=InteriorPointSolver.Mode.MAXIMIZE,
                                 c=[9, 10, 16, 0, 0, 0],
                                 a=[
                                     [18, 15, 12,1,0,0],
                                     [6, 4, 8,0,1,0],
                                     [5, 3, 3,0,0,1]
                                 ],
                                 b=[360, 192, 180],
                                 start=[1, 1, 1, 315, 174, 169],
                                 alpha=0.9,
                                 eps=5)

    solver.solve()

    print()
    print("Simplex: ")
    solver = SimplexSolver(mode=SimplexSolver.Mode.MAXIMIZE,
                           c=[9, 10, 16],
                           a=[
                               [18, 15, 12],
                               [6, 4, 8],
                               [5, 3, 3]
                           ],
                           b=[360, 192, 180],
                           eps=5)

    solver.solve()


    print()
    print("--> Example 2. An unbounded objective function")

    print("Intetior point alpha = 0,5: ")
    solver = InteriorPointSolver(mode=InteriorPointSolver.Mode.MAXIMIZE,
                                 c=[5, 4, 0, 0],
                                 a=[
                                     [1, 0,1,0],
                                     [1, -1,0,1]
                                 ],
                                 b=[7, 8],
                                 start=[1,1,6,8],
                                 alpha=0.5,
                                 eps=5)

    solver.solve()

    print()
    print("Intetior point alpha = 0,9: ")
    solver = InteriorPointSolver(mode=InteriorPointSolver.Mode.MAXIMIZE,
                                 c=[5, 4, 0, 0],
                                 a=[
                                     [1, 0,1,0],
                                     [1, -1,0,1]
                                 ],
                                 b=[7, 8],
                                 start=[1,1,6,8],
                                 alpha=0.9,
                                 eps=5)

    solver.solve()

    print()
    print("Simplex: ")
    solver = SimplexSolver(mode=SimplexSolver.Mode.MAXIMIZE,
                           c=[5, 4],
                           a=[
                               [1, 0],
                               [1, -1]
                           ],
                           b=[7, 8],
                           eps=5,)

    solver.solve()

    # print()
    # print("--> Example 3. Another unbounded function")
    # simplex_solve_and_check(
    #     mode=SimplexSolver.Mode.MAXIMIZE,
    #     objective_function=[5, 4],
    #     constraints_matrix=[
    #         [1, -1],
    #         [1, 0]
    #     ],
    #     constraints_right_hand_side=[8, 7],
    #     epsilon=5,
    #     expected_solution=None
    # )
    #
    # print()
    # print("--> Example 4")
    # simplex_solve_and_check(
    #     mode=SimplexSolver.Mode.MAXIMIZE,
    #     objective_function=[1, 2, 3],
    #     constraints_matrix=[
    #         [1, 1, 1],
    #         [2, 1, 1],
    #     ],
    #     constraints_right_hand_side=[10, 20],
    #     epsilon=5,
    #     expected_solution=30
    # )
    #
    # print()
    # print("--> Example 5 - An unbounded function. Taken from lab 5.1, task 2")
    # simplex_solve_and_check(
    #     mode=SimplexSolver.Mode.MAXIMIZE,
    #     objective_function=[2, 3],
    #     constraints_matrix=[
    #         [4, -2],
    #         [-1, -1]
    #     ],
    #     constraints_right_hand_side=[-4, -6],
    #     epsilon=5,
    #     expected_solution=None
    # )
    #
    # print()
    print("--> Example 6 - More variables. Taken from lab 2, task 1")

    print("Intetior point alpha = 0,5: ")
    solver = InteriorPointSolver(mode=InteriorPointSolver.Mode.MAXIMIZE,
                                 c=[100, 140, 120, 0, 0, 0,0],
                                 a=[
                                     [3, 6, 7, 1, 0, 0,0],
                                     [2, 1, 8, 0, 1, 0,0],
                                     [1, 1, 1, 0, 0, 1,0],
                                     [5, 3, 3, 0, 0, 0,1]
                                 ],
                                 b=[135, 260, 220, 360],
                                 start=[1, 1, 1, 119, 249, 117,349],
                                 alpha=0.5,
                                 eps=5)

    solver.solve()


    print("Intetior point alpha = 0,9: ")
    solver = InteriorPointSolver(mode=InteriorPointSolver.Mode.MAXIMIZE,
                                 c=[100, 140, 120, 0, 0, 0,0],
                                 a=[
                                     [3, 6, 7, 1, 0, 0,0],
                                     [2, 1, 8, 0, 1, 0,0],
                                     [1, 1, 1, 0, 0, 1,0],
                                     [5, 3, 3, 0, 0, 0,1]
                                 ],
                                 b=[135, 260, 220, 360],
                                 start=[1, 1, 1, 119, 249, 117,349],
                                 alpha=0.9,
                                 eps=5)

    solver.solve()


    print("Simplex: ")
    solver = SimplexSolver(
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
    )

    solver.solve()

    print()
    print("--> Example 7 - Minimization, taken from tutorial")
    print("Intetior point alpha = 0,5: ")
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


    print("Intetior point alpha = 0,9: ")
    solver = InteriorPointSolver(mode=InteriorPointSolver.Mode.MINIMIZE,
                                 c=[-2, 2, -6, 0, 0, 0],
                                 a=[
                                     [2, 1, -2, 1, 0, 0],
                                     [1, 2, 4, 0, 1, 0],
                                     [1, -1, 2, 0, 0, 1]
                                 ],
                                 b = [24, 23, 10],
                                 start=[1, 1, 1, 23, 16, 8],
                                 alpha=0.9,
                                 eps=5)

    solver.solve()


    print("Simplex: ")
    solver = SimplexSolver(mode=InteriorPointSolver.Mode.MINIMIZE,
                           c=[-2, 2, -6],
                           a=[
                               [2, 1, -2],
                               [1, 2, 4],
                               [1, -1, 2]
                           ],
                           b=[24, 23, 10],
                           eps=5)

    solver.solve()


if __name__ == "__main__":
    main()
