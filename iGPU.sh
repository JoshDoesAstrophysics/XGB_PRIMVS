#!/bin/bash

# Default time for interactive session (minimum 8 hours)
DEFAULT_TIME="8:00:00"

# Default account and QoS (adjust according to your use case)
DEFAULT_ACCOUNT="gr-vvv"
DEFAULT_QOS="interactive"

# Function to request compute resources (Low Tier - A30)
request_old() {
    echo "Requesting Old Resource Tier: 6x P100 GPU (32GB VRAM), 64GB RAM, 8 hours"
	salloc --job-name=contrastive_gpu --time=$DEFAULT_TIME --gres=gpu:6 --cpus-per-task=64 --mem=256G --partition=teton-gpu --account=$DEFAULT_ACCOUNT --qos=$DEFAULT_QOS
}

# Function to request compute resources (Low Tier - A30)
request_low() {
    echo "Requesting Low Resource Tier: 1x A30 GPU (24GB VRAM), 64GB RAM, 8 hours"
	salloc --job-name=contrastive_gpu --time=$DEFAULT_TIME --gres=gpu:1 --cpus-per-task=64 --mem=256G --partition=mb-a30,beartooth-gpu --account=$DEFAULT_ACCOUNT --qos=$DEFAULT_QOS
}

# Function to request compute resources (Medium Tier - L40S)
request_medium() {
    echo "Requesting Medium Resource Tier: 1x L40S GPU (48GB VRAM), 128GB RAM, 8 hours"
        salloc --job-name=contrastive_gpu --time=$DEFAULT_TIME --gres=gpu:1 --cpus-per-task=64 --mem=256G --partition=mb-l40s --account=$DEFAULT_ACCOUNT --qos=$DEFAULT_QOS
}

# Function to request compute resources (High Tier - H100)
request_high() {
    echo "Requesting High Resource Tier: 1x H100 GPU (80GB VRAM), 192GB RAM, 8 hours"
	salloc --job-name=contrastive_gpu --time=$DEFAULT_TIME --gres=gpu:1 --cpus-per-task=64 --mem=256G --partition=mb-h100 --account=$DEFAULT_ACCOUNT --qos=$DEFAULT_QOS
}

# Function to request compute resources (High Tier - H100)
request_super() {
    echo "Requesting High Resource Tier: 3x H100 GPU (80GB VRAM), 256GB RAM, 8 hours"
	salloc --job-name=contrastive_gpu --time=$DEFAULT_TIME --gres=gpu:4 --cpus-per-task=64 --mem=256G --partition=mb-h100 --account=$DEFAULT_ACCOUNT --qos=$DEFAULT_QOS
}

# Function to request compute resources (High Tier - H100)
request_max() {
    echo "Requesting High Resource Tier: 8x H100 GPU (80GB VRAM), 1228G RAM, 8 hours"
	salloc --job-name=contrastive_gpu --time=$DEFAULT_TIME --gres=gpu:8 --cpus-per-task=64 --mem=1228G --partition=mb-h100 --account=$DEFAULT_ACCOUNT --qos=$DEFAULT_QOS
}

# Function to request compute resources (Test Tier - A6000)
request_test() {
    echo "Requesting Test Resource Tier: 4x A6000 GPU (48GB VRAM), 256GB RAM, 8 hours"
	salloc --job-name=contrastive_gpu --time=$DEFAULT_TIME --gres=gpu:4 --cpus-per-task=64 --mem=256G --partition=mb-a6000 --account=$DEFAULT_ACCOUNT --qos=$DEFAULT_QOS
}

Usage message
usage() {
    echo "Usage: $0 {old|low|medium|high}"
    echo "old    - Request old resources (1x P100 GPU, 64GB RAM, 8 hours)"
    echo "low    - Request low-tier resources (1x A30 GPU, 64GB RAM, 8 hours)"
    echo "medium - Request medium-tier resources (1x L40S GPU, 128GB RAM, 8 hours)"
    echo "high   - Request high-tier resources (1x H100 GPU, 256GB RAM, 8 hours)"
    echo "super  - Request super-tier resources (3x H100 GPU, 256GB RAM, 8 hours)"
    echo "max    - Request max-tier resources (6x H100 GPU, 256GB RAM, 8 hours)"
    echo "test   - Request test-tier resources (1x A6000 GPU, 256GB RAM, 8 hours)"
}

Main logic to choose tier
if [ $# -ne 1 ]; then
   usage
	exit 1
fi

case $1 in
    old)
        request_old
        ;;
    low)
        request_low
        ;;
    medium)
        request_medium
        ;;
    high)
        request_high
        ;;
    super)
        request_super
        ;;
    max)
        request_max
        ;;
    test)
        request_test
        ;;
    *)
        usage
   	exit 1
   	;;
esac
