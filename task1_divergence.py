import numpy as np
import time
from numba import cuda

@cuda.jit
def uniform_kernel(d_in, d_out):
    idx = cuda.grid(1)

    if idx < d_in.size:
        value = d_in[idx]

        for j in range(1000):
            value = value * 1.0001 + 0.0001

        d_out[idx] = value

@cuda.jit
def divergent_kernel(d_in, d_out):
    idx = cuda.grid(1)

    if idx < d_in.size:
        value = d_in[idx]

        if idx % 2 == 0:
            for j in range(1000):
                value = value * 1.0001 + 0.0001
        else:
            for j in range(1000):
                value = (value - 0.0001) / 1.0001

        d_out[idx] = value

@cuda.jit
def warp_aligned_kernel(d_in, d_out):
    idx = cuda.grid(1)

    if idx < d_in.size:
        value = d_in[idx]
        warp_id = idx // 32

        if warp_id % 2 == 0:
            for j in range(1000):
                value = value * 1.0001 + 0.0001
        else:
            for j in range(1000):
                value = (value - 0.0001) / 1.0001

        d_out[idx] = value

N = 1_000_000

h_in = np.ones(N, dtype=np.float32)
d_in = cuda.to_device(h_in)
d_out = cuda.device_array_like(d_in)

threads_per_block = 256
blocks_per_grid = (N + threads_per_block - 1) // threads_per_block

def benchmark_kernel(kernel, name):
    # Warm-up
    kernel[blocks_per_grid, threads_per_block](d_in, d_out)
    cuda.synchronize()

    times = []

    for trial in range(10):
        start = time.perf_counter()

        kernel[blocks_per_grid, threads_per_block](d_in, d_out)
        cuda.synchronize()

        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    avg_time = sum(times) / len(times)

    print(f"{name}: {avg_time:.4f} ms")
    return avg_time

time_a = benchmark_kernel(uniform_kernel, "Kernel A - Uniform")
time_b = benchmark_kernel(divergent_kernel, "Kernel B - Divergent")
time_c = benchmark_kernel(warp_aligned_kernel, "Kernel C - Warp Aligned")

print("\nRESULTS")
print(f"Uniform:      {time_a:.4f} ms")
print(f"Divergent:    {time_b:.4f} ms")
print(f"Warp-Aligned: {time_c:.4f} ms")
