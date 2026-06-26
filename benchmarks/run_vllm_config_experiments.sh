#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

MODEL_PATH="${MODEL_PATH:-/workspace/genrecall/models/Qwen2.5-0.5B-Instruct}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
METRICS_URL="${METRICS_URL:-http://127.0.0.1:${PORT}/metrics}"
MODELS_URL="${MODELS_URL:-http://127.0.0.1:${PORT}/v1/models}"
DTYPE="${DTYPE:-auto}"

MAX_MODEL_LEN_VALUES="${MAX_MODEL_LEN_VALUES:-4096}"
MAX_NUM_BATCHED_TOKENS_VALUES="${MAX_NUM_BATCHED_TOKENS_VALUES:-2048}"
MAX_NUM_SEQS_VALUES="${MAX_NUM_SEQS_VALUES:-64}"
GPU_MEMORY_UTILIZATION_VALUES="${GPU_MEMORY_UTILIZATION_VALUES:-0.8}"
BLOCK_SIZE_VALUES="${BLOCK_SIZE_VALUES:-16}"
KV_CACHE_DTYPE_VALUES="${KV_CACHE_DTYPE_VALUES:-auto}"

LOAD_CMD="${LOAD_CMD:-}"
WARMUP_CMD="${WARMUP_CMD:-}"
BENCH_NUM_PROMPTS="${BENCH_NUM_PROMPTS:-1000}"
BENCH_REQUEST_RATE="${BENCH_REQUEST_RATE:-inf}"
BENCH_MAX_CONCURRENCY="${BENCH_MAX_CONCURRENCY:-}"
BENCH_RANDOM_INPUT_LEN="${BENCH_RANDOM_INPUT_LEN:-1024}"
BENCH_RANDOM_OUTPUT_LEN="${BENCH_RANDOM_OUTPUT_LEN:-128}"
BENCH_RANDOM_PREFIX_LEN="${BENCH_RANDOM_PREFIX_LEN:-0}"
BENCH_RANDOM_RANGE_RATIO="${BENCH_RANDOM_RANGE_RATIO:-0.0}"
BENCH_NUM_WARMUPS="${BENCH_NUM_WARMUPS:-0}"
BENCH_EXTRA_ARGS="${BENCH_EXTRA_ARGS:-}"
RUN_SECONDS="${RUN_SECONDS:-0}"
WARMUP_SECONDS="${WARMUP_SECONDS:-10}"
COOLDOWN_SECONDS="${COOLDOWN_SECONDS:-300}"
SERVER_READY_TIMEOUT_SECONDS="${SERVER_READY_TIMEOUT_SECONDS:-300}"
START_AT="${START_AT:-}"
RESULT_DIR="${RESULT_DIR:-$ROOT_DIR/benchmarks/results/vllm_config_$(date +%Y%m%d_%H%M%S)}"
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"
DRY_RUN="${DRY_RUN:-0}"

usage() {
  cat <<'EOF'
Usage:
  run_vllm_config_experiments.sh [options]

Options:
  --max-model-len VALUES             Comma-separated values. Default: 4096.
  --max-num-batched-tokens VALUES    Comma-separated values. Default: 2048.
  --max-num-seqs VALUES              Comma-separated values. Default: 64.
  --gpu-memory-utilization VALUES    Comma-separated values. Default: 0.8.
  --block-size VALUES                Comma-separated values. Default: 16.
  --kv-cache-dtype VALUES            Comma-separated values. Default: auto.
  --start-at TIME                    Start time. Supports HH:MM or YYYY-mm-dd HH:MM.
  --load-cmd CMD                     Command to run for each experiment.
                                     If omitted, uses "vllm bench serve".
  --warmup-cmd CMD                   Optional warmup command before LOAD_CMD.
  --bench-num-prompts N              vLLM bench --num-prompts. Default: 1000.
  --bench-request-rate R             vLLM bench --request-rate. Default: inf.
  --bench-max-concurrency N          vLLM bench --max-concurrency.
  --bench-random-input-len N         vLLM bench --random-input-len. Default: 1024.
  --bench-random-output-len N        vLLM bench --random-output-len. Default: 128.
  --bench-random-prefix-len N        vLLM bench --random-prefix-len. Default: 0.
  --bench-random-range-ratio R       vLLM bench --random-range-ratio. Default: 0.0.
  --bench-num-warmups N              vLLM bench --num-warmups. Default: 0.
  --bench-extra-args ARGS            Extra args appended to vLLM bench serve.
  --run-seconds N                    Deprecated fallback. vLLM bench is used by default.
  --warmup-seconds N                 Sleep after server ready before load. Default: 10.
  --cooldown-seconds N               Gap after each experiment. Default: 300.
  --result-dir DIR                   Where logs/manifest are written.
  --extra-vllm-args ARGS             Extra args appended to vLLM serve.
  --dry-run                          Print experiment matrix only.
  -h, --help                         Show this help.

Environment variables with the same uppercase names are also supported.

Examples:
  START_AT=23:00 \
  ./benchmarks/run_vllm_config_experiments.sh \
    --max-model-len 2048,4096 \
    --max-num-seqs 16,64 \
    --gpu-memory-utilization 0.5,0.8 \
    --bench-num-prompts 1000 \
    --bench-request-rate inf \
    --bench-max-concurrency 64 \
    --cooldown-seconds 300

Notes:
  The script expands a Cartesian product of the specified value lists.
  If an option is omitted, the script uses the single default value listed above.
  If an option is provided, only the user-provided comma-separated values are used.
  By default, each run executes vLLM's online serving benchmark:
  vllm bench serve --dataset-name random --save-result ...
EOF
}

while (($#)); do
  case "$1" in
    --max-model-len) MAX_MODEL_LEN_VALUES="$2"; shift 2 ;;
    --max-num-batched-tokens) MAX_NUM_BATCHED_TOKENS_VALUES="$2"; shift 2 ;;
    --max-num-seqs) MAX_NUM_SEQS_VALUES="$2"; shift 2 ;;
    --gpu-memory-utilization) GPU_MEMORY_UTILIZATION_VALUES="$2"; shift 2 ;;
    --block-size) BLOCK_SIZE_VALUES="$2"; shift 2 ;;
    --kv-cache-dtype) KV_CACHE_DTYPE_VALUES="$2"; shift 2 ;;
    --start-at) START_AT="$2"; shift 2 ;;
    --load-cmd) LOAD_CMD="$2"; shift 2 ;;
    --warmup-cmd) WARMUP_CMD="$2"; shift 2 ;;
    --bench-num-prompts) BENCH_NUM_PROMPTS="$2"; shift 2 ;;
    --bench-request-rate) BENCH_REQUEST_RATE="$2"; shift 2 ;;
    --bench-max-concurrency) BENCH_MAX_CONCURRENCY="$2"; shift 2 ;;
    --bench-random-input-len) BENCH_RANDOM_INPUT_LEN="$2"; shift 2 ;;
    --bench-random-output-len) BENCH_RANDOM_OUTPUT_LEN="$2"; shift 2 ;;
    --bench-random-prefix-len) BENCH_RANDOM_PREFIX_LEN="$2"; shift 2 ;;
    --bench-random-range-ratio) BENCH_RANDOM_RANGE_RATIO="$2"; shift 2 ;;
    --bench-num-warmups) BENCH_NUM_WARMUPS="$2"; shift 2 ;;
    --bench-extra-args) BENCH_EXTRA_ARGS="$2"; shift 2 ;;
    --run-seconds) RUN_SECONDS="$2"; shift 2 ;;
    --warmup-seconds) WARMUP_SECONDS="$2"; shift 2 ;;
    --cooldown-seconds) COOLDOWN_SECONDS="$2"; shift 2 ;;
    --result-dir) RESULT_DIR="$2"; shift 2 ;;
    --extra-vllm-args) EXTRA_VLLM_ARGS="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

split_values() {
  local raw="$1"
  local default_if_empty="$2"
  if [[ -z "$raw" ]]; then
    printf '%s\n' "$default_if_empty"
  else
    local old_ifs="$IFS"
    IFS=','
    read -r -a values <<< "$raw"
    IFS="$old_ifs"
    local value
    for value in "${values[@]}"; do
      value="${value#"${value%%[![:space:]]*}"}"
      value="${value%"${value##*[![:space:]]}"}"
      [[ -n "$value" ]] && printf '%s\n' "$value"
    done
  fi
}

wait_until_start_at() {
  local start_at="$1"
  [[ -z "$start_at" ]] && return 0

  local target_epoch now_epoch
  now_epoch="$(date +%s)"

  if [[ "$start_at" =~ ^[0-9]{1,2}:[0-9]{2}$ ]]; then
    target_epoch="$(date -d "today $start_at" +%s)"
    if (( target_epoch <= now_epoch )); then
      target_epoch="$(date -d "tomorrow $start_at" +%s)"
    fi
  else
    target_epoch="$(date -d "$start_at" +%s)"
  fi

  local wait_seconds=$((target_epoch - now_epoch))
  if (( wait_seconds > 0 )); then
    printf '[%s] Waiting until %s (%s seconds)\n' \
      "$(date '+%F %T')" "$(date -d "@$target_epoch" '+%F %T')" "$wait_seconds"
    sleep "$wait_seconds"
  fi
}

port_is_open() {
  python3 - "$PORT" <<'PY'
import socket
import sys

port = int(sys.argv[1])
sock = socket.socket()
sock.settimeout(0.5)
try:
    sock.connect(("127.0.0.1", port))
except OSError:
    sys.exit(1)
else:
    sys.exit(0)
finally:
    sock.close()
PY
}

wait_for_server_ready() {
  local deadline=$((SECONDS + SERVER_READY_TIMEOUT_SECONDS))
  while (( SECONDS < deadline )); do
    if curl --noproxy '*' -fsS "$MODELS_URL" >/dev/null 2>&1 \
      || curl --noproxy '*' -fsS "$METRICS_URL" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  return 1
}

stop_server() {
  local pid="${1:-}"
  [[ -z "$pid" ]] && return 0
  if kill -0 "$pid" >/dev/null 2>&1; then
    printf '[%s] Stopping vLLM server pid=%s\n' "$(date '+%F %T')" "$pid"
    kill "$pid" >/dev/null 2>&1 || true
    local i
    for i in {1..30}; do
      if ! kill -0 "$pid" >/dev/null 2>&1; then
        return 0
      fi
      sleep 1
    done
    printf '[%s] vLLM server did not stop gracefully; killing pid=%s\n' "$(date '+%F %T')" "$pid"
    kill -9 "$pid" >/dev/null 2>&1 || true
  fi
}

declare -a max_model_lens max_num_batched_tokens_list max_num_seqs_list
declare -a gpu_memory_utilization_list block_size_list kv_cache_dtype_list

mapfile -t max_model_lens < <(split_values "$MAX_MODEL_LEN_VALUES" "4096")
mapfile -t max_num_batched_tokens_list < <(split_values "$MAX_NUM_BATCHED_TOKENS_VALUES" "2048")
mapfile -t max_num_seqs_list < <(split_values "$MAX_NUM_SEQS_VALUES" "64")
mapfile -t gpu_memory_utilization_list < <(split_values "$GPU_MEMORY_UTILIZATION_VALUES" "0.8")
mapfile -t block_size_list < <(split_values "$BLOCK_SIZE_VALUES" "16")
mapfile -t kv_cache_dtype_list < <(split_values "$KV_CACHE_DTYPE_VALUES" "auto")

mkdir -p "$RESULT_DIR"
MANIFEST="$RESULT_DIR/manifest.tsv"

printf 'run_id\tstart_time\tend_time\tstatus\tmax_model_len\tmax_num_batched_tokens\tmax_num_seqs\tgpu_memory_utilization\tblock_size\tkv_cache_dtype\tlog_file\n' > "$MANIFEST"

declare -a matrix=()
for max_model_len in "${max_model_lens[@]}"; do
  for max_num_batched_tokens in "${max_num_batched_tokens_list[@]}"; do
    for max_num_seqs in "${max_num_seqs_list[@]}"; do
      for gpu_memory_utilization in "${gpu_memory_utilization_list[@]}"; do
        for block_size in "${block_size_list[@]}"; do
          for kv_cache_dtype in "${kv_cache_dtype_list[@]}"; do
            matrix+=("$max_model_len|$max_num_batched_tokens|$max_num_seqs|$gpu_memory_utilization|$block_size|$kv_cache_dtype")
          done
        done
      done
    done
  done
done

printf 'Experiment count: %s\n' "${#matrix[@]}"
printf 'Result dir: %s\n' "$RESULT_DIR"
printf 'Manifest: %s\n' "$MANIFEST"
printf 'Cooldown seconds: %s\n' "$COOLDOWN_SECONDS"
printf 'Warmup seconds: %s\n' "$WARMUP_SECONDS"
if [[ -n "$START_AT" ]]; then
  printf 'Start at: %s\n' "$START_AT"
fi
if [[ -n "$LOAD_CMD" ]]; then
  printf 'Load command: %s\n' "$LOAD_CMD"
else
  printf 'Load command: vllm bench serve, random dataset\n'
  printf '  bench num prompts: %s\n' "$BENCH_NUM_PROMPTS"
  printf '  bench request rate: %s\n' "$BENCH_REQUEST_RATE"
  printf '  bench max concurrency: %s\n' "${BENCH_MAX_CONCURRENCY:-<unset>}"
  printf '  bench random input len: %s\n' "$BENCH_RANDOM_INPUT_LEN"
  printf '  bench random output len: %s\n' "$BENCH_RANDOM_OUTPUT_LEN"
  printf '  bench random prefix len: %s\n' "$BENCH_RANDOM_PREFIX_LEN"
  printf '  bench random range ratio: %s\n' "$BENCH_RANDOM_RANGE_RATIO"
  printf '  bench num warmups: %s\n' "$BENCH_NUM_WARMUPS"
  printf '  bench extra args: %s\n' "${BENCH_EXTRA_ARGS:-<none>}"
fi
if [[ -z "$LOAD_CMD" && "$RUN_SECONDS" != "0" ]]; then
  printf 'Note: RUN_SECONDS=%s is ignored because vLLM bench is the default load command.\n' "$RUN_SECONDS"
fi

run_index=0
for item in "${matrix[@]}"; do
  IFS='|' read -r max_model_len max_num_batched_tokens max_num_seqs gpu_memory_utilization block_size kv_cache_dtype <<< "$item"
  run_index=$((run_index + 1))
  printf '  %03d max_model_len=%s max_num_batched_tokens=%s max_num_seqs=%s gpu_memory_utilization=%s block_size=%s kv_cache_dtype=%s\n' \
    "$run_index" "$max_model_len" "$max_num_batched_tokens" "$max_num_seqs" \
    "$gpu_memory_utilization" "$block_size" "$kv_cache_dtype"
done

if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi

wait_until_start_at "$START_AT"

run_index=0
for item in "${matrix[@]}"; do
  IFS='|' read -r max_model_len max_num_batched_tokens max_num_seqs gpu_memory_utilization block_size kv_cache_dtype <<< "$item"
  run_index=$((run_index + 1))
  run_id="$(printf 'run_%03d_len%s_batchtok%s_seqs%s_gpu%s_block%s_kv%s' \
    "$run_index" "$max_model_len" "$max_num_batched_tokens" "$max_num_seqs" \
    "$gpu_memory_utilization" "$block_size" "$kv_cache_dtype" | tr '/: ' '___')"
  log_file="$RESULT_DIR/${run_id}.log"
  status="ok"
  start_time="$(date '+%F %T')"

  printf '\n[%s] Starting %s\n' "$start_time" "$run_id" | tee -a "$log_file"

  if port_is_open; then
    printf 'Error: port %s is already open. Stop the existing vLLM server before running experiments.\n' "$PORT" | tee -a "$log_file" >&2
    status="port_busy"
    end_time="$(date '+%F %T')"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$run_id" "$start_time" "$end_time" "$status" "$max_model_len" \
      "$max_num_batched_tokens" "$max_num_seqs" "$gpu_memory_utilization" \
      "$block_size" "$kv_cache_dtype" "$log_file" >> "$MANIFEST"
    exit 1
  fi

  export MODEL_PATH SERVED_MODEL_NAME HOST PORT DTYPE
  export MAX_MODEL_LEN="$max_model_len"
  export MAX_NUM_SEQS="$max_num_seqs"
  export MAX_NUM_BATCHED_TOKENS="$max_num_batched_tokens"
  export GPU_MEMORY_UTILIZATION="$gpu_memory_utilization"
  export BLOCK_SIZE="$block_size"
  export KV_CACHE_DTYPE="$kv_cache_dtype"

  (
    set -x
    "$SCRIPT_DIR/start_vllm_server.sh" $EXTRA_VLLM_ARGS
  ) >> "$log_file" 2>&1 &
  server_pid=$!
  trap 'stop_server "$server_pid"' EXIT INT TERM

  if ! wait_for_server_ready; then
    printf '[%s] Server did not become ready in %s seconds\n' \
      "$(date '+%F %T')" "$SERVER_READY_TIMEOUT_SECONDS" | tee -a "$log_file" >&2
    status="server_not_ready"
  else
    printf '[%s] Server ready. Metrics: %s\n' "$(date '+%F %T')" "$METRICS_URL" | tee -a "$log_file"
    sleep "$WARMUP_SECONDS"

    if [[ -n "$WARMUP_CMD" ]]; then
      printf '[%s] Running warmup command\n%s\n' "$(date '+%F %T')" "$WARMUP_CMD" | tee -a "$log_file"
      if ! bash -lc "$WARMUP_CMD" >> "$log_file" 2>&1; then
        status="warmup_failed"
      fi
      sleep "$WARMUP_SECONDS"
    fi

    if [[ "$status" == "ok" && -n "$LOAD_CMD" ]]; then
      export VLLM_EXPERIMENT_RUN_ID="$run_id"
      export VLLM_EXPERIMENT_INDEX="$run_index"
      export VLLM_EXPERIMENT_LOG="$log_file"
      printf '[%s] Running load command\n%s\n' "$(date '+%F %T')" "$LOAD_CMD" | tee -a "$log_file"
      if ! bash -lc "$LOAD_CMD" >> "$log_file" 2>&1; then
        status="load_failed"
      fi
    elif [[ "$status" == "ok" ]]; then
      bench_result_dir="$RESULT_DIR/vllm_bench"
      mkdir -p "$bench_result_dir"
      bench_cmd=(
        vllm bench serve
        --backend openai
        --host 127.0.0.1
        --port "$PORT"
        --model "$MODEL_PATH"
        --served-model-name "$SERVED_MODEL_NAME"
        --tokenizer "$MODEL_PATH"
        --dataset-name random
        --random-input-len "$BENCH_RANDOM_INPUT_LEN"
        --random-output-len "$BENCH_RANDOM_OUTPUT_LEN"
        --random-prefix-len "$BENCH_RANDOM_PREFIX_LEN"
        --random-range-ratio "$BENCH_RANDOM_RANGE_RATIO"
        --num-prompts "$BENCH_NUM_PROMPTS"
        --request-rate "$BENCH_REQUEST_RATE"
        --num-warmups "$BENCH_NUM_WARMUPS"
        --percentile-metrics ttft,tpot,itl,e2el
        --metric-percentiles 50,95,99
        --request-id-prefix "${run_id}-"
        --save-result
        --result-dir "$bench_result_dir"
        --result-filename "${run_id}.json"
        --label "$run_id"
        --metadata
        "run_id=$run_id"
        "max_model_len=$max_model_len"
        "max_num_batched_tokens=$max_num_batched_tokens"
        "max_num_seqs=$max_num_seqs"
        "gpu_memory_utilization=$gpu_memory_utilization"
        "block_size=$block_size"
        "kv_cache_dtype=$kv_cache_dtype"
      )
      if [[ -n "$BENCH_MAX_CONCURRENCY" ]]; then
        bench_cmd+=(--max-concurrency "$BENCH_MAX_CONCURRENCY")
      fi
      if [[ -n "$BENCH_EXTRA_ARGS" ]]; then
        read -r -a bench_extra_args <<< "$BENCH_EXTRA_ARGS"
        bench_cmd+=("${bench_extra_args[@]}")
      fi
      printf '[%s] Running vLLM benchmark\n' "$(date '+%F %T')" | tee -a "$log_file"
      printf '  result: %s/%s.json\n' "$bench_result_dir" "$run_id" | tee -a "$log_file"
      printf '  command:' | tee -a "$log_file"
      printf ' %q' "${bench_cmd[@]}" | tee -a "$log_file"
      printf '\n' | tee -a "$log_file"
      if ! "${bench_cmd[@]}" >> "$log_file" 2>&1; then
        status="vllm_bench_failed"
      fi
    fi
  fi

  stop_server "$server_pid"
  trap - EXIT INT TERM
  end_time="$(date '+%F %T')"
  printf '[%s] Finished %s status=%s\n' "$end_time" "$run_id" "$status" | tee -a "$log_file"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$run_id" "$start_time" "$end_time" "$status" "$max_model_len" \
    "$max_num_batched_tokens" "$max_num_seqs" "$gpu_memory_utilization" \
    "$block_size" "$kv_cache_dtype" "$log_file" >> "$MANIFEST"

  if (( run_index < ${#matrix[@]} )); then
    printf '[%s] Cooldown for %s seconds before next run\n' "$(date '+%F %T')" "$COOLDOWN_SECONDS" | tee -a "$log_file"
    sleep "$COOLDOWN_SECONDS"
  fi
done

printf '\nAll experiments finished. Manifest: %s\n' "$MANIFEST"
