# Use NVIDIA PyTorch container as base image
FROM nvcr.io/nvidia/pytorch:25.01-py3

ARG CUDA_VERSION=12.8.1
ENV CUDA_VERSION=${CUDA_VERSION}
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update \
    && apt-get update \
    && apt install -y software-properties-common \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated ca-certificates \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
    wget git \
    curl \
    build-essential \
    gcc-11 g++-11 \
    libgl1-mesa-dev \
    libglib2.0-0 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

RUN pip install --ignore-installed kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu128.html

WORKDIR /workspace
COPY . .

RUN pip install -r requirements.txt
RUN pip install -e .
RUN pip install "flash_attn @ https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
