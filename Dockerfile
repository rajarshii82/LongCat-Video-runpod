FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CHECKPOINT_DIR=/runpod-volume/weights/LongCat-Video

WORKDIR /app

RUN apt-get update && \
    apt-get install -y git ffmpeg libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY . /app

RUN pip install --no-cache-dir ninja packaging psutil

RUN MAX_JOBS=4 pip install --no-cache-dir \
    flash-attn==2.7.4.post1 --no-build-isolation

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir runpod

CMD ["python", "-u", "handler.py"]