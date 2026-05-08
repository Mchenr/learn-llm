#!/bin/bash

SIZES="512 1024 2048 4096"
ITERS=10
RESULTS_DIR="results"

mkdir -p $RESULTS_DIR

echo "=========================================="
echo "  Matmul Benchmark - Progressive Learning"
echo "=========================================="
echo ""

make clean && make

echo ""
echo "Running benchmarks..."
echo ""

for SIZE in $SIZES; do
    echo "--- Matrix ${SIZE}x${SIZE} ---"
    for target in 01_naive_cuda 02_tiled_cuda 03_regtiling_cuda 04_wmma_cuda 05_cublas; do
        if [ -f "$target" ]; then
            echo ""
            ./$target $SIZE $SIZE $SIZE $ITERS 2>&1 | tee -a $RESULTS_DIR/${target}_${SIZE}.log
        fi
    done
    echo ""
done

echo ""
echo "=========================================="
echo "  Benchmark complete. Results in $RESULTS_DIR/"
echo "=========================================="
