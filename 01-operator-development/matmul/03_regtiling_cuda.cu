#include "common.h"

#define BM 64
#define BN 64
#define BK 32
#define TM 4
#define TN 4

__global__ void matmul_regtiling_kernel(const float* A, const float* B, float* C,
                                        int M, int N, int K) {
    __shared__ float As[BK][BM + 1];
    __shared__ float Bs[BK][BN + 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * (BN / TN) + tx;

    int row = blockIdx.y * BM + ty * TM;
    int col = blockIdx.x * BN + tx * TN;

    float accum[TM][TN] = {0};

    int num_tiles = (K + BK - 1) / BK;
    for (int t = 0; t < num_tiles; t++) {
        for (int i = tid; i < BM * BK; i += BM / TM * BN / TN) {
            int m = i % BM;
            int k = i / BM;
            int g_row = blockIdx.y * BM + m;
            int g_col = t * BK + k;
            As[k][m] = (g_row < M && g_col < K) ? A[g_row * K + g_col] : 0.0f;
        }

        for (int i = tid; i < BK * BN; i += BM / TM * BN / TN) {
            int n = i % BN;
            int k = i / BN;
            int g_row = t * BK + k;
            int g_col = blockIdx.x * BN + n;
            Bs[k][n] = (g_row < K && g_col < N) ? B[g_row * N + g_col] : 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < BK; k++) {
            float a_reg[TM], b_reg[TN];
            for (int m = 0; m < TM; m++)
                a_reg[m] = As[k][ty * TM + m];
            for (int n = 0; n < TN; n++)
                b_reg[n] = Bs[k][tx * TN + n];
            for (int m = 0; m < TM; m++)
                for (int n = 0; n < TN; n++)
                    accum[m][n] += a_reg[m] * b_reg[n];
        }

        __syncthreads();
    }

    for (int m = 0; m < TM; m++)
        for (int n = 0; n < TN; n++) {
            int c_row = row + m;
            int c_col = col + n;
            if (c_row < M && c_col < N)
                C[c_row * N + c_col] = accum[m][n];
        }
}

int main(int argc, char** argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 1024;
    int N = (argc > 2) ? atoi(argv[2]) : 1024;
    int K = (argc > 3) ? atoi(argv[3]) : 1024;
    int iters = (argc > 4) ? atoi(argv[4]) : 10;

    printf("[RegTiling CUDA] M=%d, N=%d, K=%d, BM=%d, BN=%d, BK=%d, TM=%d, TN=%d\n",
           M, N, K, BM, BN, BK, TM, TN);

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

    dim3 blockDim(BN / TN, BM / TM);
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

    GpuTimer timer;
    timer.start();
    matmul_regtiling_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    timer.stop();
    CUDA_CHECK(cudaDeviceSynchronize());
    float warmup_ms = timer.elapsed_ms();

    float total_ms = 0.0f;
    for (int i = 0; i < iters; i++) {
        timer.start();
        matmul_regtiling_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
        timer.stop();
        total_ms += timer.elapsed_ms();
    }
    float avg_ms = total_ms / iters;
    double gflops = calc_gflops(M, N, K, avg_ms);

    printf("  Grid: (%d, %d), Block: (%d, %d)\n", gridDim.x, gridDim.y,
           blockDim.x, blockDim.y);
    printf("  Shared memory per block: %.1f KB\n",
           ((BK * (BM + 1) + BK * (BN + 1)) * sizeof(float)) / 1024.0);
    printf("  Elements per thread: %d (TM=%d x TN=%d)\n", TM * TN, TM, TN);
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
