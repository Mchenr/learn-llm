#include "common.h"
#include <mma.h>
using namespace nvcuda::wmma;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define WARP_SIZE 32

__global__ void matmul_wmma_kernel(const half* A, const half* B, float* C,
                                   int M, int N, int K) {
    int warpM = blockIdx.y;
    int warpN = blockIdx.x;

    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, col_major> b_frag;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    fill_fragment(c_frag, 0.0f);

    int num_tiles = (K + WMMA_K - 1) / WMMA_K;
    for (int t = 0; t < num_tiles; t++) {
        int a_row = warpM * WMMA_M;
        int a_col = t * WMMA_K;
        int b_row = t * WMMA_K;
        int b_col = warpN * WMMA_N;

        if (a_row < M && a_col < K)
            load_matrix_sync(a_frag, A + a_row * K + a_col, K);
        else
            fill_fragment(a_frag, __float2half(0.0f));

        if (b_row < K && b_col < N)
            load_matrix_sync(b_frag, B + b_row * N + b_col, N);
        else
            fill_fragment(b_frag, __float2half(0.0f));

        mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    int c_row = warpM * WMMA_M;
    int c_col = warpN * WMMA_N;
    if (c_row < M && c_col < N)
        store_matrix_sync(C + c_row * N + c_col, c_frag, N, mem_row_major);
}

int main(int argc, char** argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 1024;
    int N = (argc > 2) ? atoi(argv[2]) : 1024;
    int K = (argc > 3) ? atoi(argv[3]) : 1024;
    int iters = (argc > 4) ? atoi(argv[4]) : 10;

    if (M % WMMA_M != 0 || N % WMMA_N != 0 || K % WMMA_K != 0) {
        printf("Error: M, N, K must be multiples of %d for WMMA\n", WMMA_M);
        return 1;
    }

    printf("[WMMA CUDA] M=%d, N=%d, K=%d, WMMA tile=%dx%dx%d\n", M, N, K,
           WMMA_M, WMMA_N, WMMA_K);

    size_t bytes_A = M * K * sizeof(half);
    size_t bytes_B = K * N * sizeof(half);
    size_t bytes_C = M * N * sizeof(float);

    half *h_A = (half*)malloc(bytes_A);
    half *h_B = (half*)malloc(bytes_B);
    float *h_C = (float*)malloc(bytes_C);

    init_matrix_half(h_A, M * K);
    init_matrix_half(h_B, K * N);

    float *h_A_f = (float*)malloc(M * K * sizeof(float));
    float *h_B_f = (float*)malloc(K * N * sizeof(float));
    for (int i = 0; i < M * K; i++) h_A_f[i] = __half2float(h_A[i]);
    for (int i = 0; i < K * N; i++) h_B_f[i] = __half2float(h_B[i]);

    half *d_A, *d_B;
    float *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
    CUDA_CHECK(cudaMalloc(&d_C, bytes_C));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));

    dim3 gridDim(N / WMMA_N, M / WMMA_M);
    dim3 blockDim(WARP_SIZE);

    GpuTimer timer;
    timer.start();
    matmul_wmma_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    timer.stop();
    CUDA_CHECK(cudaDeviceSynchronize());
    float warmup_ms = timer.elapsed_ms();

    float total_ms = 0.0f;
    for (int i = 0; i < iters; i++) {
        timer.start();
        matmul_wmma_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
        timer.stop();
        total_ms += timer.elapsed_ms();
    }
    float avg_ms = total_ms / iters;
    double gflops = calc_gflops(M, N, K, avg_ms);

    printf("  Grid: (%d, %d), Block: %d (1 warp)\n", gridDim.x, gridDim.y,
           blockDim.x);
    printf("  Precision: FP16 input, FP32 accumulation\n");
    printf("  Warmup: %.3f ms\n", warmup_ms);
    printf("  Average: %.3f ms, GFLOPS: %.2f\n", avg_ms, gflops);

    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));
    float* ref = (float*)malloc(bytes_C);
    matmul_cpu_ref(h_A_f, h_B_f, ref, M, N, K);
    verify_result(h_C, ref, M * N);
    free(ref);

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_A_f);
    free(h_B_f);
    return 0;
}
