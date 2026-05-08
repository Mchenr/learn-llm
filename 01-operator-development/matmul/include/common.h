#pragma once
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                   \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

inline void init_matrix(float* mat, int size) {
    srand(42);
    for (int i = 0; i < size; i++)
        mat[i] = (float)(rand() % 100) / 100.0f;
}

inline void init_matrix_half(half* mat, int size) {
    srand(42);
    for (int i = 0; i < size; i++)
        mat[i] = __float2half((float)(rand() % 100) / 100.0f);
}

inline void matmul_cpu_ref(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
                sum += A[i * K + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
}

inline bool verify_result(const float* C, const float* ref, int size, float tol = 5e-2f) {
    float max_err = 0.0f;
    int max_err_idx = 0;
    for (int i = 0; i < size; i++) {
        float err = fabsf(C[i] - ref[i]);
        if (err > max_err) {
            max_err = err;
            max_err_idx = i;
        }
    }
    printf("  Max error: %e (at index %d)\n", max_err, max_err_idx);
    if (max_err > tol) {
        printf("  FAILED: max error %e > tolerance %e\n", max_err, tol);
        return false;
    }
    printf("  PASSED\n");
    return true;
}

inline double calc_gflops(int M, int N, int K, double ms) {
    return 2.0 * (double)M * (double)N * (double)K / (ms * 1e6);
}

class GpuTimer {
public:
    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~GpuTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    void start() { CUDA_CHECK(cudaEventRecord(start_)); }
    void stop() { CUDA_CHECK(cudaEventRecord(stop_)); }
    float elapsed_ms() {
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

private:
    cudaEvent_t start_, stop_;
};
