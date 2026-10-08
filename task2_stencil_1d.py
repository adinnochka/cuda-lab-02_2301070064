import numpy as np
from numba import cuda

@cuda.jit
def stencil_1d(d_in, d_out, N):
    idx = cuda.grid(1)

    if idx < N:
        left = d_in[idx - 1] if idx > 0 else d_in[0]
        center = d_in[idx]
        right = d_in[idx + 1] if idx < N - 1 else d_in[N - 1]

        d_out[idx] = 0.25 * left + 0.5 * center + 0.25 * right

def cpu_stencil(arr):
    padded = np.pad(arr, (1, 1), mode="edge")
    return 0.25 * padded[:-2] + 0.5 * padded[1:-1] + 0.25 * padded[2:]

def run_stencil(h_in):
    N = len(h_in)

    d_in = cuda.to_device(h_in)
    d_out = cuda.device_array_like(d_in)

    threads_per_block = 256
    blocks_per_grid = (N + threads_per_block - 1) // threads_per_block

    stencil_1d[blocks_per_grid, threads_per_block](d_in, d_out, N)
    cuda.synchronize()

    return d_out.copy_to_host()


