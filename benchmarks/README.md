# Benchmarks

本目录保存硬件信息采集、推理服务启动和 vLLM 参数实验脚本。脚本默认围绕单卡 RTX 4090D 的本地推理 benchmark 使用，运行前需要确认当前 shell 已进入对应 CUDA / Python / vLLM 环境。

## 文件说明

| 文件 | 作用 | 主要输出 |
| --- | --- | --- |
| `collect_hardware_info.sh` | 采集系统、NVIDIA GPU、CUDA 工具链、Python 包版本 | `benchmarks/hardware_info_*.txt` |
| `hardware_baseline.md` | 第一周硬件与软件基线记录模板 | 手动回填后的基线文档 |
| `start_vllm_server.sh` | 按统一参数启动 OpenAI-compatible vLLM server | 前台 vLLM 服务日志 |
| `run_vllm_config_experiments.sh` | 展开 vLLM 参数矩阵，逐组启动 server 并运行 `vllm bench serve` | `benchmarks/results/vllm_config_*/` |
| `start_prometheus.sh` | 用 Docker 启动单独 Prometheus 容器抓取 vLLM metrics | Prometheus UI 和本地 TSDB 数据 |

`benchmarks/results/` 已在 `.gitignore` 中忽略，适合存放批量实验日志和 JSON 结果。

## 1. 采集硬件和软件环境

```bash
bash benchmarks/collect_hardware_info.sh
```

指定输出文件：

```bash
bash benchmarks/collect_hardware_info.sh benchmarks/hardware_info_local.txt
```

脚本会尽量执行这些检查：

- 系统、CPU、内存信息。
- `nvidia-smi`、GPU 显存、driver、compute capability、功耗和频率信息。
- `nvcc`、Nsight Systems、Nsight Compute 版本。
- Python、pip、PyTorch CUDA、Triton、Transformers、vLLM 版本。

采集完成后，把关键结果回填到 `hardware_baseline.md`。如果某些命令不存在，脚本会记录 `command not found`，不会中断整体采集。

## 2. 启动 vLLM Server

```bash
MODEL_PATH=/path/to/model \
SERVED_MODEL_NAME=local-model \
bash benchmarks/start_vllm_server.sh
```

常用环境变量：

| 变量 | 默认值 | 含义 |
| --- | --- | --- |
| `MODEL_PATH` | `/workspace/genrecall/models/Qwen2.5-0.5B-Instruct` | 模型目录或模型名 |
| `SERVED_MODEL_NAME` | `Qwen/Qwen2.5-0.5B-Instruct` | OpenAI API 中暴露的模型名 |
| `HOST` | `0.0.0.0` | 服务监听地址 |
| `PORT` | `8000` | 服务端口 |
| `DTYPE` | `auto` | vLLM 加载精度 |
| `MAX_MODEL_LEN` | `4096` | 最大上下文长度 |
| `MAX_NUM_SEQS` | `16` | 最大并发序列数 |
| `MAX_NUM_BATCHED_TOKENS` | `2048` | 单轮调度最大 token 数 |
| `GPU_MEMORY_UTILIZATION` | `0.8` | vLLM 可使用的 GPU 显存比例 |
| `BLOCK_SIZE` | `16` | KV cache block size |
| `KV_CACHE_DTYPE` | `auto` | KV cache 数据类型 |

脚本会检查 `vllm` 命令和 `MODEL_PATH` 是否存在。额外 vLLM 参数可以直接追加在命令末尾：

```bash
bash benchmarks/start_vllm_server.sh --enable-prefix-caching
```

## 3. 批量运行 vLLM 参数实验

先用 `--dry-run` 检查实验矩阵：

```bash
bash benchmarks/run_vllm_config_experiments.sh \
  --max-model-len 2048,4096 \
  --max-num-seqs 16,64 \
  --gpu-memory-utilization 0.6,0.8 \
  --dry-run
```

正式运行：

```bash
MODEL_PATH=/path/to/model \
SERVED_MODEL_NAME=local-model \
bash benchmarks/run_vllm_config_experiments.sh \
  --max-model-len 2048,4096 \
  --max-num-batched-tokens 1024,2048 \
  --max-num-seqs 16,64 \
  --gpu-memory-utilization 0.6,0.8 \
  --bench-num-prompts 1000 \
  --bench-request-rate inf \
  --bench-max-concurrency 64 \
  --bench-random-input-len 1024 \
  --bench-random-output-len 128
```

脚本会对下列参数做笛卡尔积：

- `max_model_len`
- `max_num_batched_tokens`
- `max_num_seqs`
- `gpu_memory_utilization`
- `block_size`
- `kv_cache_dtype`

每组实验流程：

1. 检查目标端口是否已被占用。
2. 使用 `start_vllm_server.sh` 启动 vLLM server。
3. 等待 `/v1/models` 或 `/metrics` 可访问。
4. 默认运行 `vllm bench serve --dataset-name random`。
5. 停止 server，记录状态，等待 cooldown 后进入下一组。

结果目录默认形如：

```text
benchmarks/results/vllm_config_YYYYmmdd_HHMMSS/
  manifest.tsv
  run_*.log
  vllm_bench/
    run_*.json
```

`manifest.tsv` 记录每组参数、开始时间、结束时间、状态和日志路径。`vllm_bench/*.json` 是 vLLM benchmark 保存的结构化结果，适合后续汇总 TTFT、TPOT、ITL、E2E latency 和吞吐。

## 4. 自定义压测命令

如果不想使用默认 `vllm bench serve`，可以传入自定义命令：

```bash
LOAD_CMD='python3 my_load_test.py --base-url http://127.0.0.1:8000/v1' \
bash benchmarks/run_vllm_config_experiments.sh
```

可选 warmup：

```bash
WARMUP_CMD='curl -s http://127.0.0.1:8000/v1/models >/dev/null' \
LOAD_CMD='python3 my_load_test.py' \
bash benchmarks/run_vllm_config_experiments.sh
```

运行时会暴露这些环境变量给 `LOAD_CMD`：

- `VLLM_EXPERIMENT_RUN_ID`
- `VLLM_EXPERIMENT_INDEX`
- `VLLM_EXPERIMENT_LOG`

## 5. 启动 Prometheus

```bash
bash benchmarks/start_prometheus.sh
```

默认配置：

| 变量 | 默认值 |
| --- | --- |
| `CONTAINER_NAME` | `prometheus` |
| `PROMETHEUS_IMAGE` | `prom/prometheus` |
| `PROMETHEUS_CONFIG` | `/home/c00913906/genrecall/prometheus/prometheus.yml` |
| `PROMETHEUS_DATA_DIR` | `/home/c00913906/genrecall/prometheus/data` |
| `PROMETHEUS_RETENTION` | `15d` |

这个脚本使用 Docker `--network host`，用于让 Prometheus 直接抓取宿主机的 `http://127.0.0.1:8000/metrics`。如果使用本仓库 `monitoring/` 里的 compose 方案，优先参考 `monitoring/README.md`。

## 6. 推荐记录方式

每次 benchmark 至少记录：

- 模型、精度、上下文长度、输入输出长度。
- `max_model_len`、`max_num_batched_tokens`、`max_num_seqs`、`gpu_memory_utilization`、`block_size`、`kv_cache_dtype`。
- TTFT、TPOT / ITL、E2E latency、tokens/s。
- 峰值显存、KV cache 使用率、GPU 利用率。
- 观察到的瓶颈和下一步实验假设。

建议把原始日志留在 `benchmarks/results/`，把筛选后的关键结论整理到 `reports/weekly/` 或对应框架笔记中。
