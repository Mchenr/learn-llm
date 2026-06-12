# vLLM Architecture Map

目标：在不部署 vLLM 的前提下，先建立模块地图。知道一次请求在 vLLM 里经过哪些组件，以及 scheduler、KV cache manager、worker 分别决定什么。

## 1. 推荐参考资料

按顺序读：

1. vLLM Architecture Overview
  [https://docs.vllm.ai/en/stable/design/arch_overview.html](https://docs.vllm.ai/en/stable/design/arch_overview.html)
2. vLLM Block Manager API
  [https://docs.vllm.ai/en/v0.9.0/api/vllm/core/block_manager.html](https://docs.vllm.ai/en/v0.9.0/api/vllm/core/block_manager.html)
3. vLLM V1 Scheduler API
  [https://docs.vllm.ai/en/v0.13.0/api/vllm/v1/core/sched/scheduler/](https://docs.vllm.ai/en/v0.13.0/api/vllm/v1/core/sched/scheduler/)
4. PagedAttention 论文
  [https://arxiv.org/abs/2309.06180](https://arxiv.org/abs/2309.06180)
5. vLLM 技术报告
  [https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-192.pdf](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-192.pdf)

今天优先读 1、2、3。论文和技术报告只用来补背景。

## 2. 请求路径

先用自己的话补全这条链路：

```text
client request
-> OpenAI-compatible API server
-> engine
-> scheduler
-> KV cache manager / block manager
-> worker
-> model executor
-> attention backend
-> logits processor / sampler
-> response stream
```

API server 负责处理 OpenAI- compatible HTTP 请求，完成 OpenAI schema 校验、message 到 prompt 的转换、tokenization、SamplingParams 构造、request id 分配，随后通过 ZMQ 协议发送到 engine core；engine core 负责调度、请求状态管理（?）、 KV Cache 管理，生成本轮 model execution request；worker 负责在对应 device 执行模型 forward，attention backend 根据 block table 读取 KV Cache，得到 logits；sampler 选择 next token，结果再由 engine 包装成一次性响应或者 stream chuck 返回。

## 3. 模块职责表

| 模块                               | 负责什么                                                        | 输入                                | 输出                               | 今天需要弄懂的问题                                                                                                    |
| -------------------------------- | ----------------------------------------------------------- | --------------------------------- | -------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| API server                       | 接收 OpenAI-compatible HTTP 请求，校验参数，转换 messages/prompt，构造内部请求 | HTTP request                      | Request object / stream response | OpenAI-compatible API 到内部 request 如何转换？协议层转换，包含校验、字段解析、message 转成 prompt，构造 SamplingParams 对象、分配 request id |
| Engine                           | 负责 schedule、KV Cache 管理                                     | request / scheduler output        | model execution request          | engine 是协调者，不是执行者                                                                                            |
| Scheduler                        | TODO                                                        | waiting / running requests        | 本轮要执行的 tokens                    | 它如何决定 prefill、decode、batching？                                                                               |
| KV cache manager / block manager | TODO                                                        | request token 需求                  | block allocation / block ids     | 没有空闲 block 时怎么办？                                                                                             |
| Worker                           | 负责执行模型 forward pass                                         | execution request                 | logits / sampled token           | vLLM V1 通常 一个 worker 控制一个 GPU 进程，worker 总数受并行配置的影响                                                           |
| Model executor                   | TODO                                                        | input ids / positions / KV blocks | hidden states / logits           | 它如何调用模型 forward？                                                                                             |
| Attention backend                | TODO                                                        | Q/K/V + block table               | attention output                 | PagedAttention kernel 如何读取非连续 KV blocks？                                                                     |
| Sampler                          | TODO                                                        | logits                            | next token ids                   | sampling 是否可能成为 decode 瓶颈？                                                                                   |

## 4. Scheduler 需要重点理解的事

今天不要陷入所有代码细节，只抓这些问题：

- 一个请求什么时候进入 waiting 队列？

情况一：当请求被 stop，如果请求是 resumable，后续还有新的输入或等待新的输入，会被加入 waiting 队列；

情况二：当新请求被加入 schedule 时；

情况三：当 RUNNING 请求被抢占时；

- 一个请求什么时候进入 running 队列？

条件一：waiting 或 skipped_waiting 还有请求，且 token_budget 还有剩余；

条件二：不超过 max_num_running_reqs；

条件三：请求在非 block 状态；

条件四：资源充足，比如 LoRA 不超出数量限制；

条件五：KV Cache block 分配成功；

- prefill 请求和 decode 请求能否在同一轮调度中混合

可以，默认开启 chuck prefill 调度，优先调度 decode 请求，如果 max_num_batched_tokens 没有填满，就把 prefill 请求切分过来填充。

- `max_num_batched_tokens` 限制的是请求数，还是本轮 token 总数

限制了本轮 token 总数，对于 decode 请求，可以将多个请求 batch 到一起，对于 prefill 请求，可以分块组 batch。

- KV block 不够时，scheduler 是等待、抢占、swap，还是拒绝？
- continuous batching 为什么能提高 decode 阶段 GPU 利用率？

## 5. KV Cache Manager 需要重点理解的事

补全这些概念：

- free block：当前未被任何请求使用的物理 block。
- logical block：一个请求序列里的第几个 block。
- physical block：GPU KV cache 池中的实际 block。
- block table：logical block 到 physical block 的映射。
- ref count：一个物理 block 被多少请求引用。
- copy-on-write：共享 block 需要被修改时，复制出新 block。
- prefix caching：相同前缀的 KV block 可复用。

## 6. 配置项理解

| 配置                       | 直接影响                      | 过大时的问题                            | 过小时的问题                           |
| ------------------------ | ------------------------- | --------------------------------- | -------------------------------- |
| `max_model_len`          | TODO                      | TODO                              | TODO                             |
| `max_num_batched_tokens` | 影响 prefill 和 decode 请求的调度 | 改善 TTFT，prefill 占比增大，token 首时延将降低 | 改善 ITL 指标，prefill 对 decode 的影响变小 |
| `max_num_seqs`           | TODO                      | TODO                              | TODO                             |
| `gpu_memory_utilization` | TODO                      | TODO                              | TODO                             |
| `block_size`             | TODO                      | TODO                              | TODO                             |
| `kv_cache_dtype`         | TODO                      | TODO                              | TODO                             |

## 7. 今天的最小产出

完成以下内容就算 Day 3 达标：

1. 填完模块职责表。
2. 画出请求路径。
3. 写清楚 scheduler 和 KV cache manager 的关系。
4. 写清楚 `max_model_len`、`max_num_batched_tokens`、`gpu_memory_utilization` 分别影响什么。
5. 列出解除隔离后要用 vLLM 验证的 5 个配置实验。

## 8. 解除隔离后要验证的问题

待补充：

- TODO：改变 `max_model_len` 后，空闲显存和可服务并发如何变化？
- TODO：改变 `max_num_batched_tokens` 后，TTFT 和 throughput 如何变化？
- TODO：改变 `gpu_memory_utilization` 后，KV cache capacity 如何变化？
- TODO：长 prompt 请求和短 prompt 请求混合时，scheduler 行为如何变化？
- TODO：prefix 相同的请求是否能观察到 TTFT 降低？
