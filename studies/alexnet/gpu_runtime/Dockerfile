FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    binutils \
    cmake \
    libgl1-mesa-glx \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
    pip install \
    numpy \
    pytest \
    torch-tensorrt \
    torchvision

RUN mkdir -p /workspace/csrc /workspace/tests /workspace/tools

CMD ["/bin/bash"]
