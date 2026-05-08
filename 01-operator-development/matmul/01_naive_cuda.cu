#include "common.h"

__global__ void matmul_naive_kernel(const float* A, const float* B, float* C,
                                    int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++)
            sum += A[row * K + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

int main(int argc, char** argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 1024;
    int N = (argc > 2) ? atoi(argv[2]) : 1024;
    int K = (argc > 3) ? atoi(argv[3]) : 1024;
    int iters = (argc > 4) ? atoi(argv[4]) : 10;

    printf("[Naive CUDA] M=%d, N=%d, K=%d\n", M, N, K);

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

    dim3 blockDim(32, 32);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (M + blockDim.y - 1) / blockDim.y);

    GpuTimer timer;
    timer.start();
    matmul_naive_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    timer.stop();
    CUDA_CHECK(cudaDeviceSynchronize());
    float warmup_ms = timer.elapsed_ms();

    float total_ms = 0.0f;
    for (int i = 0; i < iters; i++) {
        timer.start();
        matmul_naive_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
        timer.stop();
        total_ms += timer.elapsed_ms();
    }
    float avg_ms = total_ms / iters;
    double gflops = calc_gflops(M, N, K, avg_ms);

    printf("  Grid: (%d, %d), Block: (%d, %d)\n", gridDim.x, gridDim.y,
           blockDim.x, blockDim.y);
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
