# Minimal GPU environment with Claude Code
# Build: docker build --build-arg CUDA_VERSION=12.1 -t gpu-dev .

ARG CUDA_VERSION=12.1
FROM pytorch/pytorch:2.1.0-cuda${CUDA_VERSION}-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y \
    git curl rclone sudo \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user 'dev'
RUN useradd -m -s /bin/bash dev \
    && echo "dev ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Node.js + Claude Code (installed globally as root)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/* \
    && npm install -g @anthropic-ai/claude-code

# Switch to dev user
USER dev
WORKDIR /home/dev

CMD ["bash"]
