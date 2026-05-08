#include "common.h"

#define TILE_SIZE 32

__global__ void matmul_tiled_kernel(const float* A, const float* B, float* C,
                                    int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < num_tiles; t++) {
        int a_col = t * TILE_SIZE + threadIdx.x;
        int b_row = t * TILE_SIZE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

int main(int argc, char** argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 1024;
    int N = (argc > 2) ? atoi(argv[2]) : 1024;
    int K = (argc > 3) ? atoi(argv[3]) : 1024;
    int iters = (argc > 4) ? atoi(argv[4]) : 10;

    printf("[Tiled CUDA] M=%d, N=%d, K=%d, TILE=%d\n", M, N, K, TILE_SIZE);

    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);

    float *h_A = (float*)malloc(bytes_A);
    float *h_B = (float*)malloc(bytes_B);
    float *h_C = (float*)malloc(bytes_C);

    init_matrix(h_A, M * K);
    init_matrix(h_B, K * N);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
    CUDA_CHECK(cudaMalloc(&d_C, bytes_C));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE,
                 (M + TILE_SIZE - 1) / TILE_SIZE);

    GpuTimer timer;
    timer.start();
    matmul_tiled_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    timer.stop();
    CUDA_CHECK(cudaDeviceSynchronize());
    float warmup_ms = timer.elapsed_ms();

    float total_ms = 0.0f;
    for (int i = 0; i < iters; i++) {
        timer.start();
        matmul_tiled_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
        timer.stop();
        total_ms += timer.elapsed_ms();
    }
    float avg_ms = total_ms / iters;
    double gflops = calc_gflops(M, N, K, avg_ms);

    printf("  Grid: (%d, %d), Block: (%d, %d)\n", gridDim.x, gridDim.y,
           blockDim.x, blockDim.y);
    printf("  Shared memory per block: %.1f KB\n",
           2.0 * TILE_SIZE * TILE_SIZE * sizeof(float) / 1024.0);
    printf("  Warmup: %.3f ms\n", warmup_ms);
    printf("  Average: %.3f ms, GFLOPS: %.2f\n", avg_ms, gflops);

    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));
    float* ref = (float*)malloc(bytes_C);
    matmul_cpu_ref(h_A, h_B, ref, M, N, K);
    verify_result(h_C, ref, M * N);
    free(ref);

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);
    return 0;
}
