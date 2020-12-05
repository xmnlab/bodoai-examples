import bodo


@bodo.jit
def bd_zip(np_a, np_b):
    result = []
    for i in bodo.prange(np_b.shape[0]):
        result.append((np_a[i], np_b[i]))
    return result
