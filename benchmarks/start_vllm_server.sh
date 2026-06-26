#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/workspace/genrecall/models/Qwen2.5-0.5B-Instruct}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
DTYPE="${DTYPE:-auto}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"

if ! command -v vllm >/dev/null 2>&1; then
  printf 'Error: vllm command not found. Run this script in the vLLM environment.\n' >&2
  exit 1
fi

if [[ ! -e "$MODEL_PATH" ]]; then
  printf 'Error: model path does not exist: %s\n' "$MODEL_PATH" >&2
  printf 'Set MODEL_PATH to override the default path.\n' >&2
  exit 1
fi

printf 'Starting vLLM server\n'
printf '  model: %s\n' "$MODEL_PATH"
printf '  served model name: %s\n' "$SERVED_MODEL_NAME"
printf '  endpoint: http://%s:%s\n' "$HOST" "$PORT"

cmd=(
  vllm serve "$MODEL_PATH"
  --served-model-name "$SERVED_MODEL_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --dtype "$DTYPE" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS"
)

cmd+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
cmd+=(--gpu-memory-utilization "$GPU_MEMORY_UTILIZATION")
cmd+=(--block-size "$BLOCK_SIZE")
cmd+=(--kv-cache-dtype "$KV_CACHE_DTYPE")

cmd+=("$@")

printf '  max model len: %s\n' "$MAX_MODEL_LEN"
printf '  max num seqs: %s\n' "$MAX_NUM_SEQS"
printf '  max num batched tokens: %s\n' "$MAX_NUM_BATCHED_TOKENS"
printf '  gpu memory utilization: %s\n' "$GPU_MEMORY_UTILIZATION"
printf '  block size: %s\n' "$BLOCK_SIZE"
printf '  kv cache dtype: %s\n' "$KV_CACHE_DTYPE"

exec "${cmd[@]}"
