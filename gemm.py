from torch.utils.cpp_extension import load_inline
import torch
from task import input_t, output_t

CPP_WRAPPER = """
void fp8_mm(torch::Tensor a, torch::Tensor b, torch::Tensor as, torch::Tensor bs, torch::Tensor c);
"""

CUDA_SRC = """
#include <hip/amd_detail/amd_hip_fp8.h>
#include <hip/amd_detail/amd_hip_bf16.h>

constexpr const int TILE_M = 16;
constexpr const int TILE_N = 16;
constexpr const int BLOCK = 128;

__global__ void tiled_kernel(const __hip_fp8_e4m3_fnuz* a, const __hip_fp8_e4m3_fnuz* b, const float* as, const float* bs,
                             __hip_bfloat16* c, int m, int n, int k) {
    int row = threadIdx.y + blockIdx.y * TILE_M;
    int col = threadIdx.x + blockIdx.x * TILE_N;

    if (row >= m || col >= n) return;

    __shared__ float a_tile[TILE_M][BLOCK];
    __shared__ float b_tile[TILE_N][BLOCK];

    float acc = 0.0f;

    int num_blocks_k = k / BLOCK;
    int sn = (n + BLOCK - 1) / BLOCK;

    for (int blk = 0; blk < num_blocks_k; ++blk) {
        int base_k = blk * BLOCK;

        // Load tiles from global to shared memory
        for (int i = threadIdx.x; i < BLOCK; i += TILE_N) {
            if (row < m && base_k + i < k) {
                a_tile[threadIdx.y][i] = (float)a[row + (base_k + i) * m];
            }
        }

        for (int i = threadIdx.y; i < BLOCK; i += TILE_M) {
            if (col < n && base_k + i < k) {
                b_tile[threadIdx.x][i] = (float)b[col + (base_k + i) * n];
            }
        }

        __syncthreads();

        float block_sum = 0.0f;
        for (int i = 0; i < BLOCK; ++i) {
            block_sum += a_tile[threadIdx.y][i] * b_tile[threadIdx.x][i];
        }

        __syncthreads();

        // Apply scale for this block
        float a_scale = as[row + blk * m];                     // [m x (k/BLOCK)] column-major
        float b_scale = bs[(col / BLOCK) + blk * sn];         // [(n/BLOCK) x (k/BLOCK)] row-major
        acc += block_sum * a_scale * b_scale;
    }

    // Store result
    c[row * n + col] = (__hip_bfloat16)acc; // row-major output
}

void fp8_mm(torch::Tensor a, torch::Tensor b, torch::Tensor as, torch::Tensor bs, torch::Tensor c) {
    int m = a.size(0);
    int n = b.size(0);
    int k = a.size(1);
    dim3 threads(16, 16);
    dim3 blocks((n + 15) / 16, (m + 15) / 16);

    tiled_kernel<<<blocks, threads, 0, 0>>>(
        (__hip_fp8_e4m3_fnuz*)a.data_ptr(),
        (__hip_fp8_e4m3_fnuz*)b.data_ptr(),
        as.data_ptr<float>(),
        bs.data_ptr<float>(),
        (__hip_bfloat16*)c.data_ptr(),
        m, n, k
    );
}
"""

import os
os.environ["CXX"] = "clang++"

module = load_inline(
    name='fp8_mm',
    cpp_sources=[CPP_WRAPPER],
    cuda_sources=[CUDA_SRC],
    functions=['fp8_mm'],
    verbose=True,
    extra_cuda_cflags=["--offload-arch=gfx942", "-std=c++20"],
)


def custom_kernel(data: input_t) -> output_t:
    a, b, a_scale, b_scale, c = data
    module.fp8_mm(a, b, a_scale, b_scale, c)
    return c
