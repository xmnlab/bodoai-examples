import bodo
import pandas as pd


@bodo.jit(distributed=['vec'])
def add(vec, scalar):
    return bodo.gatherv(vec + (scalar * 10))


def main():
    ix = bodo.get_rank() + 1

    data = None
    new_data = None
    new_data_t = None

    if ix == 1:
        data = pd.DataFrame({1: [1, 2, 3], 2: [4, 5, 6], 3: [7, 8, 9]})
        new_data = pd.DataFrame({1: [], 2: [], 3: []})
        new_data_t = pd.DataFrame({1: [], 2: [], 3: []})

        print('=' * 30)
        print(data)
        print('=' * 30)
        print(data.T)
        print('=' * 30)

    data = bodo.scatterv(data)
    new_data = bodo.scatterv(new_data)
    new_data_t = bodo.scatterv(new_data_t)

    if ix == 0:
        print('=' * 30)

    bodo.barrier()

    new_data = add(data, ix)
    bodo.barrier()

    if ix == 0:
        print('=' * 30)

    new_data_t = add(data.T, ix)
    bodo.barrier()

    # add.distributed_diagnostics()

    print('=' * 30)
    print(new_data)

    print('=' * 30)
    print(new_data_t)


if __name__ == '__main__':
    main()
