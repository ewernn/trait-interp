# GPU Dev Environment with Claude Code

## Setup on Remote Instance

```bash
git clone https://github.com/ewernn/trait-interp.git
cd trait-interp/docker
nvidia-smi  # check CUDA version (top right)
docker build --build-arg CUDA_VERSION=12.1 -t gpu-dev .
docker run -it --gpus all \
  -v claude_config:/home/dev/.claude \
  -v hf_cache:/home/dev/.cache/huggingface \
  -e HF_TOKEN=your_token \
  gpu-dev
```

Inside container, run `claude` and follow browser auth. Claude helps with the rest.

## What's Included

- PyTorch + CUDA (11.8, 12.1, or 12.4)
- Node.js 20 + Claude Code
- git, curl, rclone, sudo
- Non-root user `dev`
