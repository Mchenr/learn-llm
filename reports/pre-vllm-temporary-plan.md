# vLLM 部署隔离期临时学习计划

背景：当前项目环境与执行机隔离，vLLM 部署调试效率较低。执行机解除隔离前，先用几天时间完成推理优化的“引子”：建立概念框架、准备实验模板、读懂关键路径，等环境打通后可以直接进入部署和 Benchmark。

适用周期：3-5 天。

核心原则：

- 不强依赖 vLLM 成功安装。
- 不把时间消耗在远程环境反复调试上。
- 优先做解除隔离后能直接复用的准备工作。
- 每天都有一个小产出，避免只看资料。

## 临时目标

这几天结束时，应该具备：

- 能说清楚 LLM 推理的 prefill、decode、KV Cache、batching、sampling 的关系。
- 能解释 vLLM 主要解决了什么问题：PagedAttention、continuous batching、scheduler、OpenAI-compatible serving。
- 有一套待执行的 vLLM 部署检查清单。
- 有一套待运行的性能实验表格。
- 能带着明确问题去部署，而不是部署成功后才开始想测什么。

## Day 1：推理主路径梳理

主题：一次请求从 prompt 到 token stream 的完整路径。

任务：

- 阅读 `02-llm-inference/README.md` 中推理流程、性能指标、KV Cache 相关内容。
- 画出一条请求的路径：
  - tokenizer
  - prefill
  - KV Cache 写入
  - decode loop
  - sampling
  - detokenizer
  - stream output
- 整理 prefill 和 decode 的差异。

产出：

- `reports/inference-request-path.md`

必须回答：

- TTFT 主要由哪些阶段组成？
- TPOT 主要受哪些因素影响？
- 为什么 decode 比 prefill 更容易受显存带宽限制？
- KV Cache 的生命周期从哪里开始，到哪里结束？

## Day 2：KV Cache 与 PagedAttention 预习

主题：先理解 vLLM 的核心问题，再部署 vLLM。

任务：

- 推导 KV Cache 显存占用公式。
- 用一个 7B 级模型做估算：不同 batch size、sequence length、dtype 下 KV Cache 需要多少显存。
- 对比连续 KV Cache 和分页 KV Cache 的管理方式。
- 画出 block table 的概念图。

产出：

- `reports/kv-cache-memory-notes.md`

必须回答：

- KV Cache 显存和 batch size、layer 数、head 数、head dim、sequence length 的关系是什么？
- PagedAttention 优化的是注意力数学计算，还是 KV Cache 内存管理？
- 为什么服务多请求时 KV Cache 碎片会影响吞吐？

## Day 3：vLLM 架构阅读准备

主题：带着模块地图看 vLLM，而不是盲读源码。

任务：

- 整理 vLLM 的关键模块：
  - API server
  - engine
  - scheduler
  - block manager
  - worker
  - attention backend
  - model executor
- 为每个模块写一句“它负责什么”。
- 记录部署后要验证的配置项。

产出：

- `reports/vllm-architecture-map.md`

必须回答：

- scheduler 决定了什么？
- block manager 管理什么资源？
- `max_model_len`、`max_num_batched_tokens`、`gpu_memory_utilization` 分别影响什么？
- vLLM 的吞吐提升主要来自 kernel 优化、batching，还是内存管理？

## Day 4：Benchmark 方案设计

主题：先设计测什么，后面环境通了直接跑。

任务：

- 设计 vLLM 与 Transformers 的对比实验。
- 固定一组模型、prompt length、output length、并发数。
- 明确每个实验记录 TTFT、TPOT、tokens/s、显存峰值、错误日志。
- 准备结果表格。

产出：

- `benchmarks/vllm_benchmark_plan.md`

推荐实验矩阵：

| 模型 | backend | dtype | prompt length | output length | 并发 | 目标 |
|------|---------|-------|---------------|---------------|------|------|
| 0.5B / 1B | Transformers | FP16 / BF16 | 128 | 128 | 1 | 功能基线 |
| 0.5B / 1B | vLLM | FP16 / BF16 | 128 | 128 | 1 | 部署验证 |
| 7B / 8B | Transformers | FP16 / BF16 | 512 | 128 | 1 | 单请求基线 |
| 7B / 8B | vLLM | FP16 / BF16 | 512 | 128 | 1 | 单请求对比 |
| 7B / 8B | vLLM | FP16 / BF16 | 512 | 128 | 4 / 8 / 16 | continuous batching |
| 7B / 8B | vLLM | FP16 / BF16 | 2048 | 128 | 1 / 4 | 长 prompt 压力 |

必须回答：

- 哪些实验验证部署是否正确？
- 哪些实验验证 vLLM 是否带来吞吐收益？
- 哪些实验用于定位显存瓶颈？
- 哪些参数变化最可能影响 TTFT？

## Day 5：部署检查清单

主题：解除隔离后快速定位问题。

任务：

- 整理 vLLM 部署前检查项。
- 整理常见失败点和排查命令。
- 准备最小可运行命令。

产出：

- `reports/vllm-deployment-checklist.md`

检查项：

```bash
nvidia-smi
python3 --version
python3 -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
python3 -c 'import vllm; print(vllm.__version__)'
python3 -c 'import torch; print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_capability(0))'
```

最小启动命令：

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto \
  --gpu-memory-utilization 0.85
```

最小验证命令：

```bash
curl http://localhost:8000/v1/models
```

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [{"role": "user", "content": "用一句话解释 PagedAttention"}],
    "max_tokens": 64
  }'
```

## 每天复盘模板

```text
日期：
今天读了什么：
今天产出了什么：
最重要的一个概念：
还有哪些不确定：
解除隔离后要验证什么：
```

## 解除隔离后的衔接动作

执行机解除隔离后，按这个顺序继续：

1. 跑 `benchmarks/collect_hardware_info.sh`，更新 `benchmarks/hardware_baseline.md`。
2. 安装并验证 vLLM。
3. 用 0.5B / 1B 模型启动服务，确认 API 链路。
4. 用 7B / 8B 模型做单请求基线。
5. 执行并发 Benchmark，观察 continuous batching 收益。
6. 回到三个月主计划的第 1-2 周，补齐 matmul 和基础算子实验。

## 这几天不做的事

- 不在隔离环境里反复修 vLLM 安装问题。
- 不开始复杂服务化封装。
- 不做多卡并行。
- 不追求完整源码阅读，只建立模块地图。
- 不做没有记录的零散实验。
