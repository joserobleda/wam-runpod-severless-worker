# CogVideoX-5b RunPod Serverless Worker
FROM pytorch/pytorch:2.4.1-cuda11.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV WORKER_DIR=/app
ENV TORCHINDUCTOR_CACHE_DIR=/persistent_volume/torch_cache
ENV HF_HOME=/root/.cache/huggingface
ENV DIFFUSERS_CACHE=/root/.cache/huggingface/diffusers

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    git \
    curl && \
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

# Create necessary directories
RUN mkdir -p /tmp/video_generation \
    /root/.cache/huggingface/transformers \
    /root/.cache/huggingface/diffusers \
    ${WORKER_DIR}/model_cache

# Pre-download the model during build (optional - can be done at runtime)
# Uncomment the following lines if you want to download during build:
# RUN cd ${WORKER_DIR} && python3 download_model.py

# Set proper permissions
RUN chmod -R 755 ${WORKER_DIR}

CMD ["/app/startup.sh"] 