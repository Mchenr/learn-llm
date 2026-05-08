# Matmul 算子开发 - 渐进式学习

从 GPU 朴素实现到 Tensor Core 加速，逐步理解矩阵乘法的优化路径。

## 优化路径

| 步骤 | 实现 | 核心优化点 | 关键概念 |
|------|------|-----------|---------|
| 01 | Naive CUDA | GPU 并行化基线 | 1 thread = 1 element，全局内存直接访问 |
| 02 | Tiled CUDA | 共享内存分块 | Shared Memory 减少 Global Memory 访问，32×32 Tile |
| 03 | RegTiling CUDA | 寄存器分块 | 每线程算 4×4=16 元素，提高计算/访存比，Bank Conflict padding |
| 04 | WMMA CUDA | Tensor Core | WMMA API，FP16 输入 + FP32 累积，16×16×16 MMA 指令 |
| 05 | cuBLAS | 工业级参考 | cuBLAS SGEMM，代表当前 GPU 上的最优实现 |

## 各步骤详解

### 01 - Naive CUDA（基线）

将矩阵乘法直接映射到 GPU，每个线程计算 C 中的一个元素。

```
Grid: (N/32, M/32), Block: (32, 32)
每个线程: C[row][col] = Σ A[row][k] * B[k][col]
```

**问题**：每个线程独立从 Global Memory 读取 A 和 B 的行/列，大量重复访存。对于 1024×1024 矩阵，A 的每行被 1024 个线程重复读取，B 的每列也被 1024 个线程重复读取。

### 02 - Tiled CUDA

利用 Shared Memory 减少对 Global Memory 的重复访问。

- 32×32 线程块协作加载 A 和 B 的 32×32 子矩阵到 Shared Memory
- 在 Shared Memory 中完成子矩阵乘法
- 沿 K 维度迭代多个 Tile

**效果**：相比 01 通常有 2-5x 加速，Global Memory 访问减少 TILE\_SIZE 倍。

### 03 - RegTiling CUDA

在 Tiled CUDA 基础上，让每个线程计算多个 C 元素，将中间结果保存在寄存器中。

**配置**：BM=64, BN=64, BK=32, TM=4, TN=4

```
Block: 16×16 = 256 线程
每个线程: 计算 4×4 = 16 个 C 元素
Block 总计: 64×64 = 4096 个 C 元素
```

**核心变化**：

1. **寄存器累积**：每个线程持有 `float accum[4][4]`，16 个累加器全在寄存器中，零额外 Shared Memory 开销
2. **计算/访存比提升**：从 Shared Memory 读 4+4=8 个值，完成 16 次 FMA，计算访存比从 1:2 提升到 2:1
3. **Bank Conflict 消除**：Shared Memory 数组加 1 列 padding（`As[BK][BM+1]`），避免同列访问串行化

**效果**：相比 02 通常有 2-3x 加速。

### 04 - WMMA CUDA

使用 Tensor Core 的 WMMA（Warp Matrix Multiply-Accumulate）API。

- 每个 Warp（32 线程）执行一次 16×16×16 的矩阵乘加
- 输入为 FP16，累积为 FP32（混合精度）
- Tensor Core 单周期完成 16×16×16 = 4096 次 FMA

**效果**：相比 03 通常有 3-8x 加速，充分利用 4090D 的 Tensor Core。

### 05 - cuBLAS Reference

NVIDIA cuBLAS 库的 SGEMM 实现，代表当前 GPU 上的最优性能，作为其他实现的对比基准。

## 构建与运行

```bash
# 构建
make

# 运行单个实现（默认 1024x1024，10 次迭代）
./01_naive_cuda [M] [N] [K] [iterations]

# 运行全部基准测试
bash run_benchmark.sh
```

## 性能结果

> 以下结果在远程机器（NVIDIA 4090D / 鲲鹏950）上运行后填写

**硬件环境**：

- GPU: NVIDIA RTX 4090D (sm\_89, 114 SMs, 24GB VRAM)
- CPU: 鲲鹏 950 (aarch64)
- CUDA: 12.9

**GFLOPS 对比**：

| 实现 | 512×512 | 1024×1024 | 2048×2048 | 4096×4096 |
|------|---------|-----------|-----------|-----------|
| 01 Naive CUDA | 3228.00 | 4462.23 | 4519.87 | - |
| 02 Tiled CUDA | 4033.76 | 5635.90 | 5731.10 | - |
| 03 RegTiling CUDA | - | - | - | - |
| 04 WMMA CUDA | - | - | - | - |
| 05 cuBLAS | 13666.68 | 30024.99 | 44633.60 | - |

**加速比**（以 01 Naive CUDA 为基准）：

| 实现 | 512×512 | 1024×1024 | 2048×2048 | 4096×4096 |
|------|---------|-----------|-----------|-----------|
| 02 Tiled CUDA | 1.25x | 1.26x | 1.27x | - |
| 03 RegTiling CUDA | - | - | - | - |
| 04 WMMA CUDA | - | - | - | - |
| 05 cuBLAS | 4.23x | 6.73x | 9.87x | - |

## 进一步优化方向

- [x] 每个线程计算多个元素（提高计算/访存比）→ 03 RegTiling CUDA
- [ ] 双缓冲（Double Buffering）隐藏 Shared Memory 加载延迟
- [ ] 向量化访存（float4）减少加载指令数
- [ ] 使用 CUTLASS 库实现更高性能
- [ ] FP8 Tensor Core（Ada Lovelace 原生支持）
- [ ] 批量矩阵乘（Batched GEMM）用于 Attention 计算
