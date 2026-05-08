# 02 - 大模型推理基础知识

本章节系统学习大模型推理的完整流程、核心技术与优化手段。

## 学习规划

### 推理流程基础

- [ ] **自回归生成流程**
  - Prefill 阶段：并行处理 prompt tokens
  - Decode 阶段：逐 token 生成
  - KV Cache 的作用与生命周期管理

- [ ] **推理性能指标**
  - Time To First Token（TTFT）
  - Time Per Output Token（TPOT）
  - Throughput 与 Latency 的权衡
  - 内存带宽瓶颈 vs 计算瓶颈分析

### 解码策略

- [ ] **采样方法**
  - Greedy Decoding
  - Top-K / Top-P（Nucleus）采样
  - Temperature 的作用
  - Repetition Penalty 与频率惩罚

- [ ] **结构化生成**
  - Constrained Decoding
  - JSON / Grammar 约束生成
  - 逻辑约束与格式保证

### 推理加速技术

- [ ] **Speculative Decoding（投机解码）**
  - 基本原理：小模型起草 + 大模型验证
  - 自投机解码（Self-Speculative Decoding）
  - 树形投机与 Medusa
  - 接受率分析与加速比理论

- [ ] **KV Cache 优化**
  - PagedAttention（vLLM 核心技术）
  - Continuous Batching
  - KV Cache 淘汰策略（Sliding Window、H2O 等）

- [ ] **并行策略**
  - Tensor Parallelism（张量并行）
  - Pipeline Parallelism（流水线并行）
  - 数据并行与 MoE 并行

### 推理框架

- [ ] **主流推理框架对比**
  - vLLM：PagedAttention + Continuous Batching
  - TensorRT-LLM：NVIDIA 优化推理引擎
  - llama.cpp：CPU / 混合推理
  - SGLang：结构化生成优化
  - LightLLM：高性能推理框架

- [ ] **服务化部署**
  - 推理 API 设计
  - 负载均衡与弹性扩缩容
  - 多模态推理服务

## 推荐资源

- vLLM 论文与源码
- Speculative Decoding 论文系列
- 《大语言模型推理优化》相关技术博客
- SGLang 论文
