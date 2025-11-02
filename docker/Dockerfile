FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

# Base deps
RUN pip install -e .

# Optional: detection and wholebody
# RUN pip install 'wb86-pose-pipeline[detect]' 'wb86-pose-pipeline[wholebody]'

ENTRYPOINT ["wb86"]

