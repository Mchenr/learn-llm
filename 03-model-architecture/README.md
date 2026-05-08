# 03 - 大模型架构

本章节深入理解主流大模型的架构设计，从 Transformer 基础到前沿架构演进。

## 学习规划

### Transformer 基础

- [ ] **Attention 机制**
  - Scaled Dot-Product Attention 数学推导
  - Multi-Head Attention（MHA）
  - 位置编码：RoPE、ALiBi、YaRN

- [ ] **Feed-Forward Network**
  - 标准 FFN 结构
  - GLU 变体（SwiGLU、GeGLU）
  - 激活函数选择

### 主流架构分析

- [ ] **LLaMA 系列架构**
  - LLaMA / LLaMA2 / LLaMA3 架构演进
  - RMSNorm + SwiGLU + RoPE 组合
  - GQA（Grouped Query Attention）

- [ ] **其他重要架构**
  - GPT 系列：从 GPT-2 到 GPT-4
  - Mistral / Mixtral：Sliding Window + MoE
  - Qwen 系列架构
  - DeepSeek 系列：MLA（Multi-head Latent Attention）

### 架构演进方向

- [ ] **高效注意力机制**
  - Multi-Query Attention（MQA）
  - Grouped Query Attention（GQA）
  - Multi-head Latent Attention（MLA）
  - Linear Attention 与状态空间模型（Mamba）

- [ ] **MoE（Mixture of Experts）**
  - MoE 基本原理：稀疏激活
  - Router 设计与负载均衡
  - DeepSeekMoE：细粒度专家 + 共享专家
  - MoE 的训练与推理挑战

- [ ] **长上下文技术**
  - 上下文窗口扩展方法
  - RoPE 外推与内插
  - 长文本评测基准

### Tokenizer

- [ ] **分词器**
  - BPE（Byte Pair Encoding）
  - SentencePiece 与 tokenizer 训练
  - 词表大小对模型的影响

## 推荐资源

- Attention Is All You Need 论文
- LLaMA 系列论文
- Mistral / Mixtral 论文
- DeepSeek-V2 / V3 论文（MLA 架构）
