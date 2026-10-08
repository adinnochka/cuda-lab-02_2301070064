# CUDA Lab 02: Advanced Geometries & Stencils

**Student ID:** 230107064  
**Allocated GPU Node:** Tesla T4  
**CUDA Compute Capability:** 7.5  
**Official Verification Token:** A25CA9C1D7B62C7D859A — run `python verify_submission.py` on Colab GPU

## Task 1 — Warp Divergence Benchmark

Input size: 1,000,000 float32 elements; 1,000 iterations per thread; warm-up + average of 10 trials; kernel-only timings.

| Kernel | Average time (ms) |
|---|---:|
| A — Uniform | 24.1541 |
| B — Full Divergence | 90.3046 |
| C — Warp-Aligned | 44.6436 |

The interleaved divergent kernel was slowest. Warp-aligned branching was faster than interleaved branching, but slower than the uniform path.

## Task 2 — 1D Stencil

N = 10,007; `TASK 2 PASSED`; max delta = 5.960464477539063e-08.

## Task 3 — Grid-Stride Scaling

16,777,216 elements; 64 blocks × 256 threads = 16,384 GPU threads; factor = 4.25; `TASK 3 PASSED`.

## Task 4 — Sobel-X

2048 × 2048 matrix; 16 × 16 threads per block; 128 × 128 blocks; uniform input output maximum absolute value = 0.0; `TASK 4 PASSED`.

## Verification

Run `python verify_submission.py` in an environment with CUDA enabled. Paste the actual 20-character token above after verification succeeds.
