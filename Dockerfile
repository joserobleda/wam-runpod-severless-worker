# Ultra-simple Dockerfile for Diffusers approach
FROM pytorch/pytorch:2.7.0-cuda11.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV WORKER_DIR=/app
ENV TORCHINDUCTOR_CACHE_DIR=/persistent_volume/torch_cache

# Install minimal system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR ${WORKER_DIR}

# Install Python dependencies
COPY builder/requirements.txt ${WORKER_DIR}/
RUN pip install --no-cache-dir -r requirements.txt

# Copy source files
COPY src ${WORKER_DIR}/
RUN chmod +x ${WORKER_DIR}/startup.sh

# Create output and cache directories
RUN mkdir -p /tmp/video_generation /root/.cache/huggingface

CMD ["/app/startup.sh"] 