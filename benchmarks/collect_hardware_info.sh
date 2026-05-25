#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_FILE="${1:-"$ROOT_DIR/benchmarks/hardware_info_$(date +%Y%m%d_%H%M%S).txt"}"

mkdir -p "$(dirname "$OUT_FILE")"

section() {
  printf "\n## %s\n\n" "$1"
}

run_cmd() {
  local label="$1"
  shift

  printf "\n### %s\n\n" "$label"
  printf '$'
  printf ' %q' "$@"
  printf "\n\n"

  if command -v "$1" >/dev/null 2>&1; then
    "$@" 2>&1 || true
  else
    printf "command not found: %s\n" "$1"
  fi
}

{
  printf "# Hardware Info Snapshot\n"
  printf "\nGenerated at: %s\n" "$(date '+%Y-%m-%d %H:%M:%S %Z')"
  printf "Repository: %s\n" "$ROOT_DIR"

  section "System"
  run_cmd "Kernel" uname -a
  run_cmd "OS Release" sh -c 'if [ -f /etc/os-release ]; then cat /etc/os-release; elif command -v sw_vers >/dev/null 2>&1; then sw_vers; else echo "unknown"; fi'
  run_cmd "CPU" sh -c 'if command -v lscpu >/dev/null 2>&1; then lscpu; elif command -v sysctl >/dev/null 2>&1; then sysctl -n machdep.cpu.brand_string hw.ncpu; else echo "unknown"; fi'
  run_cmd "Memory" sh -c 'if command -v free >/dev/null 2>&1; then free -h; elif command -v vm_stat >/dev/null 2>&1; then vm_stat; else echo "unknown"; fi'

  section "NVIDIA GPU"
  run_cmd "GPU List" nvidia-smi -L
  run_cmd "NVIDIA SMI" nvidia-smi
  run_cmd "GPU Query" nvidia-smi --query-gpu=index,name,pci.bus_id,driver_version,memory.total,memory.used,memory.free,compute_cap,power.limit,clocks.max.graphics,clocks.max.memory --format=csv
  run_cmd "NVLink" nvidia-smi nvlink --status

  section "CUDA Toolchain"
  run_cmd "NVCC" nvcc --version
  run_cmd "CUDA Path" sh -c 'echo "CUDA_HOME=${CUDA_HOME:-}"; echo "PATH=$PATH"; echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"'
  run_cmd "Nsight Systems" nsys --version
  run_cmd "Nsight Compute" ncu --version

  section "Python Environment"
  run_cmd "Python" python3 --version
  run_cmd "Pip" python3 -m pip --version
  run_cmd "PyTorch CUDA" python3 -c 'import torch; print("torch:", torch.__version__); print("cuda available:", torch.cuda.is_available()); print("cuda runtime:", torch.version.cuda); print("device count:", torch.cuda.device_count()); [print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i), torch.cuda.get_device_properties(i).total_memory) for i in range(torch.cuda.device_count())]'
  run_cmd "Triton" python3 -c 'import triton; print("triton:", triton.__version__)'
  run_cmd "Transformers" python3 -c 'import transformers; print("transformers:", transformers.__version__)'
  run_cmd "vLLM" python3 -c 'import vllm; print("vllm:", vllm.__version__)'

  section "Suggested Next Commands"
  printf "Paste the relevant values into benchmarks/hardware_baseline.md.\n"
  printf "Then run matmul benchmarks under 01-operator-development/matmul.\n"
} | tee "$OUT_FILE"

printf "\nSaved hardware info to: %s\n" "$OUT_FILE"
