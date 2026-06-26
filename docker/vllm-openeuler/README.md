# vLLM openEuler image

This image targets the current ARM64 host and RTX 4090 D:

| Component | Version |
| --- | --- |
| Base image | `openeuler/openeuler:24.03-lts-sp3` |
| CUDA packages | `12.8` SBSA `nvcc`, CUDA runtime headers and NVRTC |
| Python | openEuler `3.11` |
| PyTorch | `2.8.0+cu128` |
| torchvision | `0.23.0+cu128` |
| torchaudio | `2.8.0+cu128` |
| vLLM | `0.11.0` |
| CUDA architecture | `8.9` |

## Build

```bash
cd /home/c00913906/genrecall/learn-llm/docker/vllm-openeuler

docker build --network=host \
  -t genrecall-vllm:0.11.0-openeuler24.03 .
```

## Verify GPU access

The image entrypoint is `vllm serve`, so override it for environment checks:

```bash
docker run --rm --gpus all \
  --entrypoint bash \
  genrecall-vllm:0.11.0-openeuler24.03 \
  -lc 'nvidia-smi && python -c "import torch, vllm; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), vllm.__version__)"'
```

## Start an OpenAI-compatible server

```bash
docker run --rm --gpus all \
  --network host \
  -v /home/c00913906/models:/models:ro \
  -v /home/c00913906/.cache/huggingface:/home/vllm/.cache/huggingface \
  genrecall-vllm:0.11.0-openeuler24.03 \
  /models/Qwen2.5-0.5B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto \
  --gpu-memory-utilization 0.85
```

The host NVIDIA driver is injected by NVIDIA Container Toolkit. Do not install
an NVIDIA kernel driver inside this image.
