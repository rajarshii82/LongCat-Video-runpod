FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CHECKPOINT_DIR=/runpod-volume/weights/LongCat-Video

ENV TORCH_CUDA_ARCH_LIST=12.0
ENV FLASH_ATTN_CUDA_ARCHS=120
ENV MAX_JOBS=4

WORKDIR /app

RUN apt-get update && \
    apt-get install -y \
    git \
    ffmpeg \
    libgl1 \
    libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY . /app

RUN pip install --no-cache-dir \
    ninja packaging psutil

RUN pip install --no-cache-dir \
    -r requirements.txt

RUN FLASH_ATTENTION_FORCE_BUILD=TRUE \
    pip install --no-cache-dir \
    flash-attn==2.8.3 \
    --no-build-isolation

RUN pip check

CMD ["python", "-u", "handler.py"]