FROM nvidia/cuda:12.9.1-devel-ubuntu22.04

# Build args
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV WORKER_MODEL_DIR=/app/model
ENV WORKER_USE_CUDA=True

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV WORKER_DIR=/app
RUN mkdir ${WORKER_DIR}
WORKDIR ${WORKER_DIR}

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL=/bin/bash
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu

# Install system dependencies
RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git sudo gcc build-essential openssh-client cmake g++ ninja-build && \
    apt-get install -y libaio-dev ffmpeg libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y python3-dev python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
                && chown -R user:user ${WORKER_DIR}
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
ENV SHELL=/bin/bash

# Create necessary directories for PyTorch/Triton optimization
RUN mkdir -p /home/user/.triton/autotune && \
    mkdir -p /home/user/.cache/matplotlib && \
    mkdir -p /home/user/.cache/huggingface && \
    mkdir -p /home/user/.cache/torch && \
    chown -R user:user /home/user/.triton /home/user/.cache

# Install build dependencies first
RUN pip install --no-cache-dir packaging>=21.0 wheel>=0.37.0 setuptools>=65.0

# Install PyTorch first (required for flash-attn build)
RUN pip install --no-cache-dir torch==2.5.1 torchaudio==2.5.1 torchvision==0.20.1

# Install Python dependencies
COPY builder/requirements.txt ${WORKER_DIR}/requirements.txt
RUN pip install --no-cache-dir -r ${WORKER_DIR}/requirements.txt && \
    rm ${WORKER_DIR}/requirements.txt

# Install git-lfs for model downloading
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
RUN sudo apt-get install git-lfs
RUN git lfs install

# Download Wan 2.2 TI2V-5B model (optimized)
RUN mkdir -p ${WORKER_MODEL_DIR}

# Download Wan 2.2 inference code first
RUN git clone https://github.com/Wan-Video/Wan2.2.git ${WORKER_DIR}/wan2.2_code

# Download Wan2.2-TI2V-5B model
RUN git clone https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B ${WORKER_MODEL_DIR}/wan2.2-ti2v-5b || \
    (echo "Git clone failed, trying with depth=1..." && \
     git clone --depth=1 https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B ${WORKER_MODEL_DIR}/wan2.2-ti2v-5b)

# Switch back to root to add files and set permissions
USER root

# Add src files and set proper permissions
ADD src ${WORKER_DIR}
RUN chown -R user:user ${WORKER_DIR} && \
    chmod +x ${WORKER_DIR}/startup.sh

# Switch back to user for runtime
USER user

# Set environment variables to reduce warnings
ENV RUNPOD_DEBUG_LEVEL=INFO
ENV MPLBACKEND=Agg
ENV TRITON_CACHE_DIR=/home/user/.triton
ENV HF_HOME=/home/user/.cache/huggingface
ENV TORCH_HOME=/home/user/.cache/torch

CMD ${WORKER_DIR}/startup.sh 