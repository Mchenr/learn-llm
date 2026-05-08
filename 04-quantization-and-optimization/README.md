# 04 - 量化与优化

本章节学习模型量化、剪枝、蒸馏等模型压缩与加速技术。

## 学习规划

### 量化基础

- [ ] **量化原理**
  - 均匀量化 vs 非均匀量化
  - 对称量化 vs 非对称量化
  - 量化粒度：Per-tensor / Per-channel / Per-group
  - 量化误差分析

- [ ] **PTQ（训练后量化）**
  - Weight-only 量化：W8A16、W4A16
  - Weight + Activation 量化：W8A8、W4A8
  - GPTQ：基于 Hessian 的逐层量化
  - AWQ：激活感知权重量化
  - QuIP / QuIP#：非均匀量化

- [ ] **QAT（量化感知训练）**
  - 伪量化（Fake Quantization）
  - Straight-Through Estimator（STE）
  - LLM-QAT：数据无关的 QAT

### 量化实践

- [ ] **主流量化方案对比**
  - GGUF（llama.cpp）：CPU 友好的多精度量化
  - GPTQ / AWQ：GPU 推理量化
  - FP8 量化：H100 / 4090 原生支持
  - INT4 / NF4（QLoRA 中的 NormalFloat）

- [ ] **量化精度评估**
  - Perplexity 评测
  - 下游任务评测
  - 量化对推理速度的实际影响

### KV Cache 量化

- [ ] **KV Cache 压缩**
  - KV8 / KV4 量化
  - Grouped KV Cache 量化
  - 对注意力质量的影响分析

### 其他压缩技术

- [ ] **剪枝（Pruning）**
  - 非结构化剪枝
  - 结构化剪枝（ShortGPT、LLM-Pruner）
  - 稀疏训练与推理

- [ ] **知识蒸馏（Knowledge Distillation）**
  - 白盒蒸馏：Logit 匹配、中间层匹配
  - 黑盒蒸馏：利用教师模型生成数据
  - MiniLLM：序列级知识蒸馏

- [ ] **低秩近似**
  - LoRA / QLoRA 推理时的合并
  - SVD 分解压缩权重

## 推荐资源

- GPTQ、AWQ 论文
- LLM.int8() 论文
- QLoRA 论文
- llama.cpp 量化方案文档
