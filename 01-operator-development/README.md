# 01 - 算子开发

本章节从底层算子入手，深入理解大模型中核心计算操作的原理与实现。

## 学习规划

### 基础算子

- [ ] **矩阵乘法（Matmul）**
  - 朴素矩阵乘法的 CPU 实现
  - 分块矩阵乘法（Tiling）优化
  - GPU CUDA 实现及共享内存优化
  - Tensor Core 的使用与 WMMA API

- [ ] **Softmax 算子**
  - 数值稳定的 Softmax 实现（减最大值技巧）
  - Online Softmax（分块计算，避免二次访存）
  - Flash Attention 中的 Softmax 实现
  - GPU 并行归约实现

- [ ] **LayerNorm / RMSNorm**
  - 计算原理与数值稳定性
  - GPU kernel 实现
  - 与 BatchNorm 的对比

### 注意力相关算子

- [ ] **Flash Attention**
  - 标准 Attention 的内存瓶颈分析
  - Flash Attention V1 核心思想：分块计算 + Online Softmax
  - Flash Attention V2 改进：并行化与减少非矩阵乘法操作
  - Flash Attention V3：利用异步与 Tensor Core

- [ ] **KV Cache 相关算子**
  - KV Cache 的内存布局（Paged Attention）
  - Prefix Cache 与 Radix Cache
  - KV Cache 量化与压缩

### 激活函数算子

- [ ] **GELU / SiLU / SwiGLU**
  - 各激活函数的数学定义与近似计算
  - GPU kernel 实现

### 通信与融合算子

- [ ] **AllReduce / AllGather 通信算子**
  - Ring AllReduce 原理
  - NCCL 的使用与调优

- [ ] **算子融合（Operator Fusion）**
  - 融合的收益分析（减少显存访问）
  - 常见融合模式：Matmul + Bias + Activation
  - 使用 Triton / CUDA 实现融合算子

## 推荐资源

- CUDA Programming Guide
- Flash Attention 论文系列
- Triton 官方教程
- OneFlow / Megatron-LM 算子实现
