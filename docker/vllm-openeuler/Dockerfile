FROM openeuler/openeuler:24.03-lts-sp3

ARG VLLM_VERSION=0.11.0
ARG TORCH_VERSION=2.8.0
ARG TORCHVISION_VERSION=0.23.0
ARG TORCHAUDIO_VERSION=2.8.0
ARG PYTORCH_CUDA_INDEX_URL=https://download.pytorch.org/whl/cu128
ARG CUDA_REPO_URL=https://developer.download.nvidia.com/compute/cuda/repos/rhel9/sbsa/cuda-rhel9.repo

LABEL org.opencontainers.image.title="vLLM on openEuler"
LABEL org.opencontainers.image.description="vLLM CUDA inference environment for openEuler 24.03 LTS SP3 ARM64"
LABEL org.opencontainers.image.source="https://github.com/vllm-project/vllm"
LABEL org.opencontainers.image.version="${VLLM_VERSION}"

ENV CUDA_HOME=/usr/local/cuda
ENV VIRTUAL_ENV=/opt/vllm
ENV PATH=/opt/vllm/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
ENV TORCH_CUDA_ARCH_LIST=8.9
ENV VLLM_TARGET_DEVICE=cuda
ENV HF_HOME=/home/vllm/.cache/huggingface
ENV VLLM_CACHE_ROOT=/home/vllm/.cache/vllm
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DEFAULT_TIMEOUT=600
ENV PIP_RETRIES=10

RUN dnf install -y --setopt=install_weak_deps=False \
        ca-certificates \
        curl \
        findutils \
        gcc \
        gcc-c++ \
        libatomic \
        libgomp \
        make \
        numactl \
        pciutils \
        procps-ng \
        python3 \
        python3-devel \
        python3-pip \
        shadow \
        which \
    && curl -fsSL "${CUDA_REPO_URL}" \
        -o /etc/yum.repos.d/cuda-rhel9-sbsa.repo \
    && dnf install -y \
        --setopt=install_weak_deps=False \
        cuda-cudart-devel-12-8 \
        cuda-nvcc-12-8 \
        cuda-nvrtc-devel-12-8 \
    && dnf clean all \
    && rm -rf /var/cache/dnf

RUN python3 -m venv "${VIRTUAL_ENV}" \
    && python -m pip install --upgrade \
        pip==25.1.1 \
        setuptools==79.0.1

RUN python -m pip install \
        --extra-index-url "${PYTORCH_CUDA_INDEX_URL}" \
        "torch==${TORCH_VERSION}" \
        "torchvision==${TORCHVISION_VERSION}" \
        "torchaudio==${TORCHAUDIO_VERSION}"

RUN python -m pip install \
        --extra-index-url "${PYTORCH_CUDA_INDEX_URL}" \
        "vllm==${VLLM_VERSION}"

RUN groupadd --gid 1000 vllm \
    && useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash vllm \
    && mkdir -p "${HF_HOME}" "${VLLM_CACHE_ROOT}" /models \
    && chown -R vllm:vllm /home/vllm /models \
    && python -c "import torch, vllm; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'vllm', vllm.__version__)" \
    && nvcc --version

USER vllm
WORKDIR /home/vllm

EXPOSE 8000

ENTRYPOINT ["vllm", "serve"]
CMD ["--help"]
