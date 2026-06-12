# 4090D Hardware Baseline

本文件记录第一周硬件与软件环境基线。先运行采集脚本，再把关键结果回填到下方表格。

```bash
bash benchmarks/collect_hardware_info.sh
```

如需指定输出文件：

```bash
bash benchmarks/collect_hardware_info.sh benchmarks/hardware_info_local.txt
```

## 1. 实验环境

| 项目                     | 值                        |
| ---------------------- | ------------------------ |
| 实验日期                   | 5.26                     |
| 主机系统                   | openEuler 24.03 SP2      |
| CPU                    | Kunpeng 950 7592C        |
| 系统内存                   | 1.5T                     |
| GPU                    | NVIDIA GeForce RTX 4090D |
| GPU 数量                 | 1                        |
| 显存容量                   | 24GB                     |
| Driver 版本              | 575.31.03                |
| CUDA Runtime 版本        | 12.9                     |
| CUDA Toolkit / NVCC 版本 | 12.9.41                  |
| cuDNN 版本               | TODO                     |
| Python 版本              | TODO                     |
| PyTorch 版本             | TODO                     |
| Triton 版本              | TODO                     |
| Transformers 版本        | TODO                     |
| vLLM 版本                | TODO                     |
| Nsight Systems 版本      | 2025.1.3.140             |
| Nsight Compute 版本      | 2025.2.0.0               |

## 2. GPU 关键规格

| 项目                 | 值                        | 备注                                           |
| ------------------ | ------------------------ | -------------------------------------------- |
| GPU 型号             | NVIDIA GeForce RTX 4090D | TODO                                         |
| Compute Capability | TODO                     | `nvidia-smi --query-gpu=compute_cap`         |
| 显存容量               | TODO                     | 约 24GB                                       |
| 显存类型               | TODO                     | GDDR6X                                       |
| 显存带宽               | TODO                     | 可填官方规格或实测带宽                                  |
| SM 数量              | TODO                     | 可填官方规格                                       |
| Tensor Core 支持     | TODO                     | FP16 / BF16 / TF32 / INT8 / INT4             |
| FP8 支持             | TODO                     | 4090D / Ada 架构需结合实际软件栈验证                     |
| 默认功耗限制             | TODO                     | `nvidia-smi`                                 |
| 最大 Graphics Clock  | TODO                     | `nvidia-smi --query-gpu=clocks.max.graphics` |
| 最大 Memory Clock    | TODO                     | `nvidia-smi --query-gpu=clocks.max.memory`   |

## 3. 软件栈确认

| 检查项               | 命令                                                            | 结果   |
| ----------------- | ------------------------------------------------------------- | ---- |
| GPU 是否可见          | `nvidia-smi -L`                                               | TODO |
| CUDA Toolkit      | `nvcc --version`                                              | TODO |
| PyTorch CUDA 是否可用 | `python3 -c 'import torch; print(torch.cuda.is_available())'` | TODO |
| PyTorch CUDA 版本   | `python3 -c 'import torch; print(torch.version.cuda)'`        | TODO |
| Triton 是否可导入      | `python3 -c 'import triton; print(triton.__version__)'`       | TODO |
| Nsight Systems    | `nsys --version`                                              | TODO |
| Nsight Compute    | `ncu --version`                                               | TODO |

## 4. Matmul 基线计划

第一轮先记录 PyTorch / cuBLAS 的矩阵乘基线，后续再和手写 CUDA kernel 对比。

| M=N=K | dtype | backend | warmup | iters | avg ms | TFLOPS | peak memory | 备注   |
| ----- | ----- | ------- | ------ | ----- | ------ | ------ | ----------- | ---- |
| 1024  | FP32  | PyTorch | TODO   | TODO  | TODO   | TODO   | TODO        | TODO |
| 2048  | FP32  | PyTorch | TODO   | TODO  | TODO   | TODO   | TODO        | TODO |
| 4096  | FP32  | PyTorch | TODO   | TODO  | TODO   | TODO   | TODO        | TODO |
| 8192  | FP32  | PyTorch | TODO   | TODO  | TODO   | TODO   | TODO        | TODO |
| 1024  | FP16  | PyTorch | TODO   | TODO  | TODO   | TODO   | TODO        | TODO |
| 2048  | FP16  | PyTorch | TODO   | TODO  | TODO   | TODO   | TODO        | TODO |
| 4096  | FP16  | PyTorch | TODO   | TODO  | TODO   | TODO   | TODO        | TODO |
| 8192  | FP16  | PyTorch | TODO   | TODO  | TODO   | TODO   | TODO        | TODO |
| 1024  | BF16  | PyTorch | TODO   | TODO  | TODO   | TODO   | TODO        | TODO |
| 2048  | BF16  | PyTorch | TODO   | TODO  | TODO   | TODO   | TODO        | TODO |
| 4096  | BF16  | PyTorch | TODO   | TODO  | TODO   | TODO   | TODO        | TODO |
| 8192  | BF16  | PyTorch | TODO   | TODO  | TODO   | TODO   | TODO        | TODO |

TFLOPS 计算：

```text
TFLOPS = 2 * M * N * K / avg_seconds / 1e12
```

## 5. 初始结论

待回填：

- 当前环境是否能稳定识别 4090D：
- PyTorch 是否能使用 CUDA：
- FP16 / BF16 是否正常：
- 后续需要补装或修复的软件：
- 第一轮性能瓶颈猜测：

## 6. 原始采集输出

采集脚本会生成 `benchmarks/hardware_info_*.txt`。这里记录文件名：

```text
TODO
```
