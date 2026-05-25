# 大模型推理优化三个月学习计划

目标：利用单张 RTX 4090D，在三个月内系统吃透大模型推理优化基础，形成可复现实验、可解释性能瓶颈、可手写关键算子、可对比主流推理框架的能力。

## 基本假设

- 硬件：单张 NVIDIA RTX 4090D，24GB 显存。
- 重点：单卡推理优化，不把多机多卡训练作为主线。
- 模型规模：以 1B、3B、7B、8B 级别模型为主，必要时使用量化模型适配 24GB 显存。
- 实践栈：CUDA C++、cuBLAS、Triton、PyTorch、Transformers、vLLM、TensorRT-LLM、llama.cpp。
- 评价方式：每个阶段都必须留下代码、实验记录、性能数据和结论，而不是只看资料。

## 总体路线

三个月分成四个阶段：

1. 第 1-3 周：GPU 性能模型与基础算子。
2. 第 4-6 周：Transformer 推理路径与 KV Cache。
3. 第 7-9 周：量化、Flash Attention、投机解码等核心优化。
4. 第 10-12 周：推理框架源码、端到端服务化 Benchmark、总结沉淀。

最终产出：

- 一套单卡 LLM 推理 Benchmark 工具。
- Matmul、Softmax、RMSNorm、Attention 关键算子的手写实现与性能对比。
- 至少两个开源推理框架的实践报告：建议 vLLM + TensorRT-LLM，补充 llama.cpp。
- 一份完整的《4090D 单卡大模型推理优化笔记》。
- 一个端到端 Demo：本地模型加载、批处理、KV Cache、量化、吞吐/延迟统计、OpenAI-compatible API。

## 每周节奏

每周固定四类任务：

- 理论：理解一个核心概念，输出笔记。
- 代码：实现或复现实验，必须能运行。
- Benchmark：记录硬件、软件版本、参数、指标和结论。
- 复盘：回答“瓶颈在哪里、为什么、下一步如何优化”。

推荐每周投入：

- 3 天读代码和文档。
- 2 天写实验代码。
- 1 天做 Benchmark 和分析。
- 1 天整理笔记与复盘。

## 指标体系

所有推理实验优先记录这些指标：

- TTFT：Time To First Token。
- TPOT：Time Per Output Token。
- Tokens/s：单请求与批量吞吐。
- 显存占用：权重、KV Cache、临时 buffer。
- GPU 利用率：SM utilization、memory throughput、tensor core utilization。
- Kernel 时间：通过 Nsight Systems / Nsight Compute / PyTorch Profiler 分析。
- 精度变化：Perplexity 或固定 prompt 输出对比。

统一记录模板：

```text
实验日期：
GPU：
驱动 / CUDA / cuDNN：
框架版本：
模型：
精度：
Prompt length：
Output length：
Batch size：
并发数：
TTFT：
TPOT：
Tokens/s：
Peak memory：
主要瓶颈：
结论：
```

## 第 1-3 周：GPU 性能模型与基础算子

目标：建立 GPU 优化的基本直觉，能解释矩阵乘为什么快或慢，知道计算瓶颈和访存瓶颈如何判断。

### 第 1 周：4090D 环境与性能基线

任务：

- 安装并确认 CUDA、NVIDIA Driver、PyTorch、Triton、Nsight Systems、Nsight Compute。
- 记录 4090D 的显存容量、显存带宽、Tensor Core 能力、CUDA capability。
- 跑通 PyTorch eager、torch.compile、cuBLAS 的矩阵乘基线。
- 建立 `benchmarks/` 目录，用统一脚本记录实验结果。

产出：

- `benchmarks/hardware_baseline.md`
- GPU 信息采集脚本。
- Matmul 基线数据：FP32、FP16、BF16，至少覆盖 1024、2048、4096、8192 尺寸。

关键问题：

- 什么时候是 compute-bound？
- 什么时候是 memory-bound？
- Roofline 模型如何指导优化？

### 第 2 周：手写 Matmul 优化

任务：

- 阅读并运行 `01-operator-development/matmul` 下已有实现。
- 对比 naive、shared memory tiling、register tiling、WMMA、cuBLAS。
- 用 Nsight Compute 分析 occupancy、global memory load efficiency、shared memory 使用、Tensor Core 使用。
- 增加一个实验表格，说明每版优化提升来自哪里。

产出：

- 更新 `01-operator-development/matmul/README.md` 的实验结果。
- 一份 Matmul 优化复盘：从 naive 到 WMMA 的性能变化。

关键问题：

- Tiling 为什么减少访存？
- register tiling 如何提高数据复用？
- WMMA 和 Tensor Core 的数据布局限制是什么？

### 第 3 周：Softmax、RMSNorm、激活函数

任务：

- 实现 CPU 与 CUDA 版本 Softmax，包含数值稳定版本。
- 实现 Online Softmax，并和普通 Softmax 对比访存次数。
- 实现 RMSNorm / LayerNorm CUDA kernel。
- 实现 SiLU / SwiGLU 的简单融合 kernel。

产出：

- `01-operator-development/softmax/`
- `01-operator-development/norm/`
- Softmax 与 RMSNorm Benchmark。

关键问题：

- 归约算子为什么常常受访存和同步限制？
- Online Softmax 如何服务 Flash Attention？
- Norm 和激活函数为什么适合 fusion？

## 第 4-6 周：Transformer 推理路径与 KV Cache

目标：把一次大模型推理拆成 Prefill、Decode、KV Cache、Sampling、Batching，并能量化每部分开销。

### 第 4 周：手写最小 Transformer 推理

任务：

- 用 PyTorch 写一个最小 decoder-only Transformer 推理脚本。
- 明确 token embedding、attention、MLP、norm、lm_head 的数据形状。
- 对比 prefill 和 decode 的计算图。
- 加入 KV Cache，并测量开启前后的速度和显存变化。

产出：

- `02-llm-inference/minimal_transformer/`
- 一份 shape tracing 笔记。

关键问题：

- Prefill 为什么更像大矩阵计算？
- Decode 为什么更容易受 memory bandwidth 限制？
- KV Cache 的显存公式是什么？

### 第 5 周：真实模型推理剖析

任务：

- 选择一个 1B-3B 模型作为日常分析模型。
- 选择一个 7B-8B 模型作为 4090D 压力测试模型。
- 使用 Transformers 跑 FP16 / BF16 推理。
- 用 profiler 拆分 attention、MLP、norm、sampling 的时间占比。
- 测试不同 prompt length、output length、batch size 的 TTFT / TPOT。

产出：

- `02-llm-inference/profiling/`
- 一份真实模型推理性能剖析报告。

关键问题：

- batch size 增大时，TTFT 和 TPOT 如何变化？
- 长上下文下显存主要被什么占用？
- attention 和 MLP 的瓶颈分别是什么？

### 第 6 周：KV Cache 与 Continuous Batching

任务：

- 手写一个简化版请求调度器，模拟多请求 decode。
- 实现固定 batch、动态 batch、continuous batching 的对比实验。
- 学习 PagedAttention 的核心思想：把 KV Cache 从连续大块变成分页管理。
- 画出 KV Cache block table 的数据结构。

产出：

- `02-llm-inference/kv-cache/`
- Continuous batching 模拟器。
- PagedAttention 原理笔记。

关键问题：

- 为什么静态 batch 会浪费 decode 计算？
- PagedAttention 解决的是计算问题还是内存管理问题？
- KV Cache 碎片如何影响吞吐？

## 第 7-9 周：核心优化技术

目标：掌握推理优化的三大抓手：减少访存、减少计算、提高批处理效率。

### 第 7 周：Flash Attention

任务：

- 从标准 attention 的中间矩阵显存占用开始推导。
- 实现一个简化版 tiled attention。
- 复现 Online Softmax 在 attention block 内的更新逻辑。
- 对比 PyTorch attention、scaled_dot_product_attention、Flash Attention。

产出：

- `01-operator-development/attention/`
- Flash Attention 原理与 Benchmark 笔记。

关键问题：

- Flash Attention 为什么不是近似算法？
- 它减少的是 HBM 访问还是 FLOPs？
- 为什么 prefill 阶段收益更明显？

### 第 8 周：量化推理

任务：

- 学习 FP16、BF16、INT8、INT4、FP8 的表示范围和硬件支持。
- 使用 GPTQ / AWQ / bitsandbytes 量化一个 7B-8B 模型。
- 对比 FP16、INT8、INT4 的显存、速度、输出质量。
- 分析 weight-only 量化为什么主要优化 decode 场景。

产出：

- `04-quantization-and-optimization/experiments/`
- 量化方案对比表。

关键问题：

- W4A16 为什么能省显存但不一定线性提速？
- group size 如何影响精度和性能？
- 量化 kernel 的 dequant 开销在哪里？

### 第 9 周：投机解码与采样优化

任务：

- 实现 greedy、top-k、top-p、temperature、repetition penalty。
- 实现一个最小 speculative decoding demo：draft model + target model。
- 测试 draft model 大小、接受率、生成长度对加速比的影响。
- 学习 constrained decoding / grammar decoding 的基本实现思路。

产出：

- `02-llm-inference/decoding/`
- 投机解码实验报告。

关键问题：

- 投机解码减少的是 target model 调用次数还是单次调用成本？
- 接受率如何决定加速比上限？
- 采样逻辑为什么也可能成为小 batch decode 的瓶颈？

## 第 10-12 周：框架源码与端到端系统

目标：把前面学过的算子、KV Cache、batching、量化放进真实推理框架和服务中理解。

### 第 10 周：vLLM 源码与实践

任务：

- 跑通 vLLM OpenAI-compatible server。
- 阅读 scheduler、block manager、attention backend 的关键路径。
- 测试不同并发、max_num_batched_tokens、max_model_len 对吞吐和延迟的影响。
- 对比 Transformers 和 vLLM 的同模型表现。

产出：

- `02-llm-inference/frameworks/vllm.md`
- vLLM 调参记录。

关键问题：

- vLLM 的吞吐优势主要来自哪里？
- scheduler 如何影响 TTFT 和 TPOT？
- PagedAttention 在代码里如何组织 KV blocks？

### 第 11 周：TensorRT-LLM / llama.cpp 对比

任务：

- 跑通 TensorRT-LLM 或至少完成 engine build 与单模型推理实验。
- 跑通 llama.cpp 的 GGUF 量化推理。
- 对比 GPU-native 框架和 CPU / mixed 推理框架的设计取舍。
- 整理 4090D 上不同框架适合的模型大小和场景。

产出：

- `02-llm-inference/frameworks/tensorrt-llm.md`
- `02-llm-inference/frameworks/llama-cpp.md`
- 框架对比矩阵。

关键问题：

- TensorRT-LLM 的优化边界在哪里？
- llama.cpp 的量化格式为什么适合本地推理？
- 框架选型应该看吞吐、延迟、部署复杂度还是模型兼容性？

### 第 12 周：端到端项目与总复盘

任务：

- 实现一个本地推理服务，支持模型加载、请求队列、batching、流式输出、指标统计。
- 支持至少两种后端：Transformers + vLLM，或 Transformers + llama.cpp。
- 整理三个月内所有 Benchmark 到统一表格。
- 写最终总结：4090D 单卡能做什么、不能做什么、优化优先级是什么。

产出：

- `projects/single-gpu-llm-server/`
- `reports/4090d-inference-optimization-summary.md`
- 一份后续学习路线。

关键问题：

- 单卡推理服务的真实性能瓶颈是什么？
- 哪些优化收益最大，哪些只是局部收益？
- 从学习项目到生产系统还缺哪些能力？

## 推荐目录演进

```text
learn-llm/
  agent.md
  benchmarks/
    hardware_baseline.md
    matmul_baseline.md
    inference_baseline.md
  reports/
    weekly/
    4090d-inference-optimization-summary.md
  projects/
    single-gpu-llm-server/
  01-operator-development/
    matmul/
    softmax/
    norm/
    attention/
  02-llm-inference/
    minimal_transformer/
    profiling/
    kv-cache/
    decoding/
    frameworks/
  04-quantization-and-optimization/
    experiments/
```

## 学习顺序原则

- 先性能模型，再算子优化。
- 先手写最小实现，再看框架源码。
- 先测 FP16 / BF16 基线，再谈量化。
- 先理解单请求，再理解 batching 和 serving。
- 先解释瓶颈，再做优化。

## 每周复盘问题

每周结束必须回答：

1. 本周跑通了什么代码？
2. 本周最重要的性能数据是什么？
3. 当前瓶颈是计算、访存、调度、显存容量还是框架开销？
4. 哪个结论是通过实验验证的？
5. 哪个问题还只是猜测？
6. 下周最应该补的实验是什么？

## 三个月验收标准

完成后应该能独立回答并演示：

- 一个 token 从输入到输出经过哪些计算步骤。
- Prefill 和 decode 为什么性能特征完全不同。
- KV Cache 如何计算显存占用，为什么需要分页管理。
- Matmul、Softmax、RMSNorm、Attention 的主要优化手段。
- Flash Attention 为什么能省显存并提速。
- INT8 / INT4 量化为什么省显存，什么时候会提速，什么时候不会。
- Continuous batching 为什么提升服务吞吐。
- vLLM、TensorRT-LLM、llama.cpp 的核心设计差异。
- 在 4090D 上如何选择模型大小、精度、batch size、上下文长度和推理框架。

## 当前第一步

从第 1 周开始：

1. 建立 `benchmarks/` 目录。
2. 写硬件信息采集脚本。
3. 跑 Matmul FP32 / FP16 / BF16 基线。
4. 把结果写入 `benchmarks/hardware_baseline.md`。
5. 再进入 `01-operator-development/matmul` 做 CUDA kernel 对比。
