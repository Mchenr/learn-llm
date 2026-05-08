#include "common.h"
#include <cublas_v2.h>

#define CUBLAS_CHECK(call)                                                      \
    do {                                                                        \
        cublasStatus_t status = call;                                           \
        if (status != CUBLAS_STATUS_SUCCESS) {                                  \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__,  \
                    status);                                                    \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

int main(int argc, char** argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 1024;
    int N = (argc > 2) ? atoi(argv[2]) : 1024;
    int K = (argc > 3) ? atoi(argv[3]) : 1024;
    int iters = (argc > 4) ? atoi(argv[4]) : 10;

    printf("[cuBLAS SGEMM] M=%d, N=%d, K=%d\n", M, N, K);

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

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    float alpha = 1.0f;
    float beta = 0.0f;

    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N));
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer timer;
    float total_ms = 0.0f;
    for (int i = 0; i < iters; i++) {
        timer.start();
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N));
        timer.stop();
        total_ms += timer.elapsed_ms();
    }
    float avg_ms = total_ms / iters;
    double gflops = calc_gflops(M, N, K, avg_ms);

    printf("  Average: %.3f ms, GFLOPS: %.2f\n", avg_ms, gflops);

    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));
    float* ref = (float*)malloc(bytes_C);
    matmul_cpu_ref(h_A, h_B, ref, M, N, K);
    verify_result(h_C, ref, M * N);
    free(ref);

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);
    return 0;
}
