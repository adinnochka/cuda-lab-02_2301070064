import numpy as np
from numba import cuda


@cuda.jit
def sobel_x_kernel(d_in, d_out, rows, cols):
    col, row = cuda.grid(2)

    if row < rows and col < cols:
        if row == 0 or row == rows - 1 or col == 0 or col == cols - 1:
            d_out[row, col] = 0.0
        else:
            gx = (
                -1.0 * d_in[row - 1, col - 1]
                + 1.0 * d_in[row - 1, col + 1]
                - 2.0 * d_in[row, col - 1]
                + 2.0 * d_in[row, col + 1]
                - 1.0 * d_in[row + 1, col - 1]
                + 1.0 * d_in[row + 1, col + 1]
            )

            d_out[row, col] = gx



def run_sobel(h_img):
    rows, cols = h_img.shape

    d_in = cuda.to_device(h_img)
    d_out = cuda.device_array_like(d_in)

    threads_2d = (16, 16)
    blocks_2d = (
        (cols + 15) // 16,
        (rows + 15) // 16
    )

    sobel_x_kernel[blocks_2d, threads_2d](
        d_in, d_out, rows, cols
    )
    cuda.synchronize()

    return d_out.copy_to_host()


