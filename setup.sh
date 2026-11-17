#!/bin/bash
set -e

echo "=== GPU Power Data Collection Campaign - Setup ==="
echo "Installing dependencies for Lambda Cloud A100 node..."

sudo apt update
sudo apt install -y \
    build-essential \
    cmake \
    git \
    htop \
    tmux \
    jq

pip install --upgrade pip

pip install \
    torch \
    torchvision \
    torchaudio \
    transformers \
    peft \
    datasets \
    accelerate \
    sentencepiece \
    diffusers[training] \
    deepspeed \
    modelscope \
    psutil \
    pandas

echo ""
echo "=== Verification ==="
echo "NVIDIA Driver:"
nvidia-smi --query-gpu=name,driver_version,power.limit --format=csv

echo ""
echo "PyTorch CUDA:"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

echo ""
echo "HuggingFace Transformers:"
python -c "import transformers; print(f'Version: {transformers.__version__}')"

echo ""
echo "DeepSpeed:"
python -c "import deepspeed; print(f'Version: {deepspeed.__version__}')"

echo ""
echo "psutil:"
python -c "import psutil; print(f'CPU count: {psutil.cpu_count()}')"

echo ""
echo "=== Setup Complete ==="
echo "Next steps:"
echo "  1. Generate configs: python generate_configs.py --num-samples 500 --gpu a100"
echo "  2. Run campaign: python run_experiment.py --config-file campaign_configs.json --gpu a100 --output-dir ./data"
echo ""
echo "For pilot test (1 session, 15 min):"
echo "  python run_experiment.py --config-file campaign_configs.json --gpu a100 --output-dir ./data --count 1"
