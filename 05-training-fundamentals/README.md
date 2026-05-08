# 05 - 大模型训练基础

本章节学习大模型训练的核心技术，从预训练到对齐的完整流程。

## 学习规划

### 预训练

- [ ] **数据工程**
  - 数据采集与清洗流程
  - 数据去重（MinHash、SimHash）
  - 数据配比与课程学习
  - Tokenizer 训练与词表设计

- [ ] **训练并行策略**
  - 数据并行（DDP、FSDP）
  - 张量并行（Megatron-LM TP）
  - 流水线并行（GPipe、1F1B）
  - 3D 并行策略组合
  - 序列并行（Sequence Parallelism）

- [ ] **训练稳定性**
  - 混合精度训练（BF16 / FP16）
  - 梯度累积与梯度裁剪
  - Loss Spike 处理
  - 学习率调度（Cosine、Warmup）

### 微调

- [ ] **全参数微调（Full Fine-tuning）**
  - 领域适配微调
  - 数据质量 vs 数据量

- [ ] **参数高效微调（PEFT）**
  - LoRA：低秩适配
  - QLoRA：量化 + LoRA
  - Adapter / Prefix-Tuning / Prompt-Tuning
  - LoRA 合并与多任务适配

- [ ] **SFT（监督微调）**
  - 指令数据构造
  - Self-Instruct 与 Evol-Instruct
  - 数据质量与多样性

### 对齐训练

- [ ] **RLHF**
  - 奖励模型训练
  - PPO 算法在大模型上的应用
  - KL 散度约束

- [ ] **DPO 及变体**
  - DPO：直接偏好优化
  - IPO / KTO / ORPO
  - 与 RLHF 的对比

- [ ] **其他对齐方法**
  - Constitutional AI（CAI）
  - RLAIF：AI 反馈强化学习
  - 安全对齐与红队测试

### 长上下文训练

- [ ] **上下文扩展训练**
  - 持续预训练扩展窗口
  - RoPE 缩放训练
  - 长文本数据的构造

## 推荐资源

- Megatron-LM 论文与源码
- LoRA / QLoRA 论文
- InstructGPT / RLHF 论文
- DPO 论文
- DeepSpeed 框架文档
