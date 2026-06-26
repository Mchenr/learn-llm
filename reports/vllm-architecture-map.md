# vLLM V1 Architecture Map

目标：以当前可运行环境为准，建立一张能被源码和实验验证的 vLLM 机制地图。先理解一次请求如何穿过系统，再逐层学习 scheduler、KV cache、model runner 和 attention backend。

## 1. 当前实验基线

更新时间：2026-06-18。

| 项目                       | 当前值                                                 |
| ------------------------ | --------------------------------------------------- |
| 容器                       | `vllm-qwen`                                         |
| 镜像                       | `vllm/vllm-openai:v0.20.0-aarch64-cu130-ubuntu2404` |
| vLLM                     | `0.20.0`，V1 Engine                                  |
| 模型                       | `Qwen2.5-0.5B-Instruct`                             |
| API                      | `http://127.0.0.1:8000`                             |
| 模型 dtype                 | `bfloat16`                                          |
| `max_model_len`          | `4096`                                              |
| `gpu_memory_utilization` | 默认 `0.92`                                           |
| `kv_cache_dtype`         | `auto`，当前随模型使用 BF16                                 |
| prefix caching           | 开启                                                  |
| chunked prefill          | 开启                                                  |
| scheduling               | asynchronous scheduling                             |
| attention backend        | FlashAttention 2                                    |
| 可用 KV Cache              | 约 `19.74 GiB`，`1,724,928 tokens`                    |
| 日志估算的 4096-token 并发      | `421.12x`，只是容量上限估算，不是实测吞吐                           |

启动命令：

```bash
vllm serve /models/Qwen2.5-0.5B-Instruct \
  --served-model-name Qwen/Qwen2.5-0.5B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto \
  --max-model-len 4096
```

当前安装源码根目录：

```text
/usr/local/lib/python3.12/dist-packages/vllm
```

本文只讨论当前 V1 实现。旧资料里的 V0 `BlockSpaceManager`、swap 等行为不能直接套用到这里。

## 2. 一次请求的完整路径

```text
HTTP request
  -> OpenAI API router / serving layer
  -> chat template + tokenizer
  -> SamplingParams + EngineCoreRequest
  -> AsyncLLM / EngineCoreClient
  -> ZMQ
  -> EngineCore
  -> Scheduler
       -> KVCacheManager 分配 block
       -> 生成 SchedulerOutput
  -> Worker / GPUModelRunner
       -> 准备 input_ids、positions、slot_mapping、block_table
       -> model.forward()
       -> attention backend 读写 paged KV cache
       -> logits + sampler
  -> EngineCoreOutput
  -> output processor + detokenizer
  -> HTTP response / SSE stream
```

需要分清两个边界：

- API 进程负责协议、模板、分词、输出流和断连处理，不决定 GPU 本轮算哪些 token。
- EngineCore 进程持有请求状态并完成调度；Worker/GPUModelRunner 才真正执行 GPU forward。

API 到内部请求的主要转换：

1. Pydantic/OpenAI schema 校验请求。
2. messages 经 chat template 转成 prompt。
3. tokenizer 把 prompt 转成 token ids。
4. OpenAI 参数转成 `SamplingParams`。
5. 构造 `EngineCoreRequest` 并分配 request id。
6. 通过 EngineCoreClient 发送给独立的 EngineCore 进程。

对应源码入口：

```text
vllm/entrypoints/openai/chat_completion/api_router.py
vllm/entrypoints/openai/chat_completion/serving.py
vllm/entrypoints/openai/chat_completion/protocol.py
vllm/v1/engine/async_llm.py
vllm/v1/engine/core_client.py
vllm/v1/engine/input_processor.py
```

## 3. 模块职责

| 模块                | 负责什么                                               | 关键输入                           | 关键输出                           |
| ----------------- | -------------------------------------------------- | ------------------------------ | ------------------------------ |
| API server        | HTTP 协议、参数校验、chat template、tokenization、流式响应       | OpenAI request                 | `EngineCoreRequest` / response |
| EngineCore        | 请求生命周期、调度循环、执行协调                                   | 新请求、上轮执行结果                     | scheduler output、engine output |
| Scheduler         | 在 token、序列数和 KV block 预算下决定本轮计算哪些请求的多少 token       | waiting/running 请求             | `SchedulerOutput`              |
| KVCacheManager    | 前缀命中查询、block 分配/释放、引用共享、缓存提交                       | token 数和 request 状态            | block ids / block table 信息     |
| Worker            | 每个设备进程的执行入口、设备与分布式环境管理                             | model execution request        | worker output                  |
| GPUModelRunner    | 准备张量和 attention metadata，调用 forward、logits、sampler | scheduler output               | sampled token ids 等            |
| Attention backend | 把新 K/V 写入槽位，并按 block table 读取历史 K/V 完成 attention   | Q/K/V、slot mapping、block table | attention output               |
| Sampler           | 对 logits 应用 temperature、top-k/top-p、惩罚和随机/贪心采样     | logits、sampling params         | next token ids                 |
| Output processor  | 合并增量 token、停止条件、detokenize、生成用户输出                  | engine core output             | `RequestOutput`                |

Sampler 通常不是普通 decode 的首要瓶颈，但在大词表、很多并发序列、请求 logprobs 或复杂采样参数时，其 logits 处理和通信成本可能变得明显，必须用 profiler 判断。

## 4. V1 Scheduler

源码：

```text
vllm/v1/core/sched/scheduler.py
```

### 4.1 prefill 和 decode 调度不互斥

从每轮调度的对象看，V1 的核心表达是 token 数量：

```text
本轮待计算 token 数
= request.num_tokens_with_spec - request.num_computed_tokens
```

- 新请求的未计算 prompt token 很多，表现为 prefill。
- 已完成 prompt 的请求通常只差下一个 token，表现为 decode。
- 长 prompt 可以受 token budget 限制而分块，表现为 chunked prefill。
- prefix caching 会直接增加 `num_computed_tokens`，跳过已命中的前缀。

### 4.2 每轮受两个主要预算约束

```text
token budget = max_num_batched_tokens
sequence budget = max_num_seqs
```

- `max_num_batched_tokens`：单次 scheduler iteration 最多处理多少 token。
- `max_num_seqs`：单次 iteration 最多允许多少条序列参与执行。

Scheduler 先处理已有的 running 请求，再尝试接纳 waiting 请求。chunked prefill 开启时，decode 和 prefill 可以混在同一个 batch 中。

### 4.3 请求状态流转

普通请求的主路径：

```text
new request
  -> WAITING
  -> RUNNING
  -> FINISHED_*
```

KV Cache 不足时可能发生：

```text
RUNNING
  -> PREEMPTED
  -> waiting queue
  -> RUNNING
```

新请求进入 waiting 队列后，只有同时满足以下条件才会进入 running：

- 本轮还有 token budget。
- 未达到 `max_num_seqs`。
- 请求未被 grammar、remote KV、streaming input 等依赖阻塞。
- LoRA、encoder 等附加资源满足约束。
- KV Cache 能成功分配所需 block。

### 4.4 KV block 不够时到底怎么办

对于 running 请求：

1. `KVCacheManager.allocate_slots()` 发现 free blocks 不足时返回 `None`。
2. Scheduler 尝试抢占 running 请求并释放其 KV blocks。
3. 被抢占请求变成 `PREEMPTED`，放回 waiting 队列。
4. 它再次被调度时，通常通过 recompute 恢复，而不是使用传统 CPU swap。
5. 如果已经没有可抢占请求，本轮停止为当前请求分配。

对于 waiting 请求：

1. `KVCacheManager.allocate_slots()` 分配失败时返回None。
2. 调度器不会因为让它运行就抢占运行中的请求，所以会放弃调度这个请求。

### 4.5 continuous batching 为什么有效

静态 batching 必须等整个 batch 中最慢的序列结束，已完成序列留下空位。

continuous batching 在每个 iteration 后移除完成请求，并立即加入 waiting 请求。不同请求可以处于不同进度，但其当前待算 token 会被动态拼成下一批，因此：

- decode 阶段不必等待整批请求同时结束。
- 新请求能更快填补空位。
- GPU 上同时活跃的序列数更稳定。
- 吞吐提高，但高负载下新请求的排队时间仍可能增加。

## 5. KV Cache、PagedAttention 与 Prefix Caching

源码：

```text
vllm/v1/core/kv_cache_manager.py
vllm/v1/core/kv_cache_coordinator.py
vllm/v1/core/block_pool.py
vllm/v1/attention/backend.py
vllm/v1/attention/ops/
```

### 5.1 关键概念

- logical block：概念上表示某条序列的第几个 token block。
- physical block：KV Cache 池中的实际存储块。
- block table：每条请求的 logical block 到 physical block 的映射。
- slot mapping：本轮每个新 token 应把 K/V 写到哪个物理槽位。
- free block：当前可重新分配的 physical block。
- ref count：多少请求或缓存引用同一个 physical block。
- prefix caching：完整 token block 的哈希相同时，共享已经计算好的 KV。
- copy-on-write：共享 block 需要被不同请求继续修改时，避免相互覆盖；具体支持和路径应以当前 backend 为准。

### 5.2 非连续 physical blocks 如何参与 attention

假设一条序列的 block table 是：

```text
logical block:   0   1   2
physical block: 17   3  41
```

历史 KV 在显存中并不连续。attention kernel 根据 token 位置计算 logical block 和块内偏移：

```text
logical_block = token_position // block_size
offset        = token_position % block_size
physical      = block_table[request, logical_block]
```

然后从 `KV_cache[physical, offset]` 读取 K/V。PagedAttention 的关键价值就是允许请求按页组织 KV，而不要求为每条序列预留一整段连续显存。

### 5.3 `block_size` 和 `hash_block_size`

- `block_size`：一个物理 KV block 存多少 token。
- `hash_block_size`：prefix caching 生成 block hash 的基础 token 粒度。
- 普通模型通常二者相同。
- 混合 KV Cache group 可能使用不同物理 block size，因此单独保留 hash 粒度。

`block_size` 小：

- 尾部碎片少，缓存匹配更细。
- block table、hash、分配回收等元数据开销更大。
- block 数更多，kernel 和管理成本可能上升。

`block_size` 大：

- block 数和元数据较少。
- 请求最后一个未填满 block 的内部碎片更严重。
- prefix cache 只能命中完整 block，复用粒度变粗。

### 5.4 Prefix caching 的边界

- 只复用完整 block；未满的尾块通常不能作为稳定前缀共享。
- 即使整个 prompt 命中，也至少要重算最后一个 token 以获得 logits。
- 它减少的是重复 prefill 计算，不直接减少后续 decode 的单 token 成本。
- 命中率高不保证端到端收益大；短 prompt、排队和网络开销可能掩盖收益。

## 6. 配置项理解

| 配置                       | 直接影响                           | 过大时的问题                                                     | 过小时的问题                                             |
| ------------------------ | ------------------------------ | ---------------------------------------------------------- | -------------------------------------------------- |
| `max_model_len`          | 单请求最大上下文；                      | <ul><li>KV Cache 压力大</li><li>并发能力下降</li></ul>                                    | <ul><li>不支持跑长 prompt</li><li>不支持长输出</li></ul>                                |
| `max_num_batched_tokens` | 影响 prefill 和 decode 请求的调度      | <ul><li>改善 TTFT</li><li>prefill 占比增大</li><li>token 首时延将降低</li></ul>                          | <ul><li>改善 ITL 指标</li><li>prefill 对 decode 的影响变小</li></ul>                   |
| `max_num_seqs`           | 单次迭代中最大 request 数              | <ul><li>TODO</li></ul>                                                       | <ul><li>TODO</li></ul>                                               |
| `gpu_memory_utilization` | 影响 GPU 分配多少显存给 KV Cache        | <ul><li>启动时可能因为显存不足而失败</li><li>容易 OOM</li></ul>                                      | <ul><li>导致 KV Cache 变小</li><li>并发能力下降</li><li>长序列可能调度不了</li><li>更容易抢占和重计算</li><li>GPU 显存浪费</li></ul> |
| `block_size`             | 每个 KV Cache block 包含的 token 数量 | <ul><li>影响前缀匹配的命中率，复用性会变差</li><li>block 尾部碎片增多，在小请求、高并发场景显存浪费明显变多</li><li>元数据管理开销低</li></ul> | <ul><li>block table 等管理开销变大</li><li>内存浪费少，前缀匹配容易命中</li></ul>                 |
| `kv_cache_dtype`         | KV Cache 量化数据类型                | <ul><li>KV Cache 显存占用高</li><li>最大并发数下降</li><li>精度和稳定性好</li></ul>                             | <ul><li>显存占用小</li><li>可容纳更长的上下文</li><li>可能引入精度损失</li><li>量化/反量化会增加计算开销</li></ul>             |

澄清与补充：

- `max_model_len`
  - “单请求最大上下文”和“调度上限”的理解是对的，它限制 prompt 与 output 的总序列长度。
  - vLLM 不会按 `max_model_len` 为每个请求提前预留一整段 KV Cache，而是从共享 block 池按需分配。
  - 调大它只是允许单请求最多占用更多 KV；是否真的降低并发，还取决于实际请求长度和 KV Cache 总容量。
  - 调低它也不一定直接增加启动日志里的 KV token capacity。
- `max_num_batched_tokens`
  - 它限制的是一次 scheduler iteration 处理的 token 总数，不是请求数。
  - “较大可能改善长 prompt 的 TTFT”和“较小有利于保护 decode ITL”这个方向基本正确，但不是固定结论。
  - 较大的 token budget 也允许更多 decode 请求一起执行，不能简单等同于“prefill 占比一定增大”。
  - 调得过小会让 prefill 被切成更多轮，可能降低 prompt throughput 并增加 TTFT。
- `max_num_seqs` 学习途径
  - 运行 `vllm serve --help=all`，先记录参数的官方定义。
  - 在 `vllm/v1/core/sched/scheduler.py` 中追踪 `max_num_running_reqs`。
  - 固定其他变量，只改变 `max_num_seqs`，记录 running、waiting、TTFT、ITL 和 throughput。
  - 实验后再回答：它限制的是“同时存在的请求总数”，还是“单次 iteration 实际参与执行的序列数”？
- `gpu_memory_utilization`
  - 它首先控制当前实例的模型执行器显存预算；权重、CUDA Graph、激活等开销占用后，剩余空间才用于 KV Cache。
  - “更容易抢占和重计算”的判断是对的，但通常发生在服务承压、KV blocks 不足时。
  - “长序列可能调度不了”要区分单序列装不下和多请求争用两种情况。
  - 它不是 GPU 算力利用率，也不是绝对的运行时显存硬上限。
- `block_size`
  - 核心取舍理解正确。
  - “容易命中”更准确地说是匹配粒度更细；实际命中率仍取决于请求之间是否存在相同 token 前缀。
  - 最终性能还受 attention backend 和硬件支持约束，不能只按显存碎片选择。
- `kv_cache_dtype`
  - 它更准确地表示 KV Cache 的存储 dtype；选择 FP8、INT8 等低位宽类型时才涉及 KV Cache 量化。
  - 它与模型权重量化相互独立，例如 W4 权重可以搭配 BF16 或 FP8 KV Cache。
  - 低位宽不一定更慢或更快，结果取决于硬件和 attention kernel。
  - `W4A8` 通常表示 4-bit 权重和 8-bit 激活；明确写 `KV8` 时才通常表示 8-bit KV Cache。
- 补充学习入口
  - 当前 v0.20.0 支持 `--kv-cache-memory-bytes`，可以用它区分“执行器显存预算”和“KV Cache 本身容量”。

## 7. 已闭环问题

### 7.1 Prefix caching 是否真的生效

实验：向当前服务连续发送两次完全相同的 1207-token prompt，每次只生成 1 token。

结果：

| 指标            | 第一次       | 第二次      |
| ------------- | --------- | -------- |
| 端到端耗时         | 约 27.9 ms | 约 9.8 ms |
| prompt tokens | 1207      | 1207     |

Prometheus 指标增量：

```text
prefix_cache_queries_total: +2414
prefix_cache_hits_total:    +1200
```

结论：

- 第二次请求命中 1200 个 token。
- 未命中的 7 个 token 来自未对齐的尾部和为取得 logits 所需的重算边界。
- 该实验同时验证了 block 粒度复用和 prefix caching 对重复 prefill 的直接收益。

可观察指标：

```bash
curl http://127.0.0.1:8000/metrics | grep -E \
  'prefix_cache_(queries|hits)|kv_cache_usage|num_requests_(running|waiting)'
```

### 7.2 KV block 不足行为

源码已经确认当前 V1 默认路径为：

```text
allocate_slots() -> None
-> scheduler preempts a running request
-> free its KV blocks
-> put it back at the front of waiting
-> recompute when admitted again
```

仍需通过缩小 KV Cache 的压力实验观察 preemption 日志、请求尾延迟和吞吐变化。

## 8. 接下来的实验矩阵

每次只改一个变量，固定模型、数据集、并发、prompt/output 长度和采样参数。记录启动日志、Prometheus 指标和客户端延迟。

| 顺序  | 实验                       | 建议对照                          | 主要观察                              |
| --- | ------------------------ | ----------------------------- | --------------------------------- |
| 1   | Prefix caching           | on/off；重复长前缀                  | hit tokens、TTFT、prompt throughput |
| 2   | `max_num_batched_tokens` | 512/1024/2048/4096            | TTFT、ITL、throughput、排队            |
| 3   | `max_num_seqs`           | 16/64/256                     | decode throughput、waiting、峰值显存    |
| 4   | KV Cache 容量              | `kv-cache-memory-bytes` 小/中/大 | cache usage、preemption、P99        |
| 5   | `kv_cache_dtype`         | BF16/FP8                      | KV token capacity、吞吐、精度           |
| 6   | `block_size`             | 平台支持的不同值                      | block 数、碎片、prefix hit、性能          |
| 7   | 长短请求混部                   | 单独运行 vs 混合                    | 短请求 TTFT/ITL、chunked prefill 行为   |
| 8   | continuous batching      | 不同并发到达模式                      | running/waiting、GPU 利用率、吞吐        |

实验结果不要只看平均值，至少记录：

- TTFT：time to first token。
- ITL/TPOT：生成阶段 token 间隔。
- E2E latency。
- prompt/output tokens per second。
- P50/P95/P99。
- running/waiting requests。
- KV cache usage、prefix hit tokens。
- preemption 次数或日志。
- GPU compute、显存和功耗。

## 9. 推荐学习顺序

### 阶段一：调度循环

目标：看懂一个请求如何从 waiting 进入 running，以及一次 `schedule()` 如何形成 token batch。

重点文件：

```text
vllm/v1/core/sched/scheduler.py
vllm/v1/core/sched/output.py
vllm/v1/request.py
```

### 阶段二：KV block 生命周期

目标：追踪请求从 prefix lookup、allocate、cache、free 到 preempt 的完整过程。

重点文件：

```text
vllm/v1/core/kv_cache_manager.py
vllm/v1/core/kv_cache_coordinator.py
vllm/v1/core/block_pool.py
```

### 阶段三：SchedulerOutput 到 GPU 输入

目标：理解 request/block 信息如何变成 `input_ids`、`positions`、`slot_mapping` 和 `block_table`。

重点文件：

```text
vllm/v1/worker/gpu_worker.py
vllm/v1/worker/gpu_model_runner.py
vllm/v1/attention/backend.py
```

### 阶段四：PagedAttention kernel

目标：从 block table 找到 physical block，并理解 K/V 写入和历史 K/V 读取。

先从当前实际使用的 FlashAttention backend 路径入手，再看通用 Triton paged attention，避免一开始陷入所有 backend。

### 阶段五：性能实验

把每个配置参数对应到：

```text
配置变化
-> scheduler/KV/model runner 的内部变化
-> metrics/logs 的变化
-> TTFT/ITL/throughput 的变化
```

这条因果链能闭环，才算真正理解一个机制。

## 10. 参考资料

优先使用与当前版本一致的源码和文档：

1. [vLLM Architecture Overview](https://docs.vllm.ai/en/stable/design/arch_overview.html)
2. [vLLM V1 Scheduler](https://docs.vllm.ai/en/stable/api/vllm/v1/core/sched/scheduler/)
3. [PagedAttention paper](https://arxiv.org/abs/2309.06180)
4. [vLLM technical report](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-192.pdf)

不要混用旧版本 Block Manager API 来解释当前 V1 行为；遇到文档和运行现象冲突时，以当前容器源码和可复现实验为准。
