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

RUN pip install --no-cache-dir \
    ninja packaging psutil

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir \
    -r requirements.txt

RUN pip install --no-cache-dir \
    flash-attn==2.8.3 \
    --no-build-isolation

RUN pip check

COPY . /app

CMD ["python", "-u", "handler.py"]