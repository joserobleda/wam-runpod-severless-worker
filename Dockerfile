# Multi-stage build for optimized image size
# Stage 1: Build stage with all tools
FROM nvidia/cuda:12.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV WORKER_DIR=/app
ENV WORKER_MODEL_DIR=/app/model

# Install build dependencies
RUN apt-get update && \
    apt-get install -y wget curl git build-essential python3-dev python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install git-lfs
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get update && \
    apt-get install -y git-lfs && \
    git lfs install

# Create directories
RUN mkdir -p ${WORKER_DIR} ${WORKER_MODEL_DIR}
WORKDIR ${WORKER_DIR}

# Install Python dependencies in builder stage
COPY builder/requirements.txt ${WORKER_DIR}/requirements.txt

# Install minimal PyTorch and dependencies
RUN pip install --no-cache-dir packaging wheel setuptools && \
    pip install --no-cache-dir torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir -r ${WORKER_DIR}/requirements.txt

# Download Wan 2.2 code (small)
RUN git clone --depth=1 https://github.com/Wan-Video/Wan2.2.git ${WORKER_DIR}/wan2.2_code && \
    rm -rf ${WORKER_DIR}/wan2.2_code/.git

# Download model with selective files only
RUN mkdir -p ${WORKER_MODEL_DIR}/wan2.2-ti2v-5b && \
    cd ${WORKER_MODEL_DIR}/wan2.2-ti2v-5b && \
    git init && \
    git remote add origin https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B && \
    git config core.sparseCheckout true && \
    echo "*.json" > .git/info/sparse-checkout && \
    echo "*.safetensors" >> .git/info/sparse-checkout && \
    echo "*.bin" >> .git/info/sparse-checkout && \
    echo "*.txt" >> .git/info/sparse-checkout && \
    git pull origin main && \
    rm -rf .git

# Stage 2: Runtime stage (much smaller base)
FROM nvidia/cuda:12.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV WORKER_DIR=/app
ENV WORKER_MODEL_DIR=/app/model
ENV WORKER_USE_CUDA=True
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Install only runtime dependencies (no build tools)
RUN apt-get update && \
    apt-get install -y python3 python3-pip ffmpeg libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN adduser --disabled-password --gecos '' --shell /bin/bash user && \
    mkdir -p ${WORKER_DIR} && \
    chown -R user:user ${WORKER_DIR}

# Copy Python environment from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code and model
COPY --from=builder ${WORKER_DIR}/wan2.2_code ${WORKER_DIR}/wan2.2_code
COPY --from=builder ${WORKER_MODEL_DIR} ${WORKER_MODEL_DIR}

# Copy source files
ADD src ${WORKER_DIR}
RUN chown -R user:user ${WORKER_DIR} && \
    chmod +x ${WORKER_DIR}/startup.sh

USER user

# Create cache directories
ENV HOME=/home/user
RUN mkdir -p /home/user/.cache/huggingface /home/user/.cache/torch /home/user/.triton

# Set environment variables
ENV RUNPOD_DEBUG_LEVEL=INFO
ENV MPLBACKEND=Agg
ENV HF_HOME=/home/user/.cache/huggingface
ENV TORCH_HOME=/home/user/.cache/torch
ENV TRITON_CACHE_DIR=/home/user/.triton

WORKDIR ${WORKER_DIR}
CMD ${WORKER_DIR}/startup.sh 