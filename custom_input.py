from interior_point_solver import InteriorPointSolver


def main():
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

    print('--> Solving with alpha = 0.5')
    InteriorPointSolver(mode, x_0, c, a, b, 0.5, eps).solve()

    print('--> Solving with alpha = 0.9')
    InteriorPointSolver(mode, x_0, c, a, b, 0.9, eps).solve()


if __name__ == '__main__':
    main()
