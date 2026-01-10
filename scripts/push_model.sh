#!/bin/bash
# Push Model to Hugging Face Hub
# ===============================
# Upload trained model to HF Hub

set -e

# Default values
MODE="combined"
REPO_NAME="crosslingual-sentiment-model"
PRIVATE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --repo-name)
            REPO_NAME="$2"
            shift 2
            ;;
        --private)
            PRIVATE=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --mode MODE           Training mode: zero_shot, low_resource, combined (default: combined)"
            echo "  --checkpoint PATH     Path to checkpoint (default: experiments/<mode>/checkpoint-best)"
            echo "  --repo-name NAME      Repository name (default: crosslingual-sentiment-model)"
            echo "  --private             Make repository private"
            echo "  --help                Show this help message"
            echo ""
            echo "Environment Variables:"
            echo "  HF_TOKEN              Hugging Face API token (required)"
            echo "  HF_USERNAME           Hugging Face username (optional, auto-detected)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check for HF_TOKEN
if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN environment variable not set"
    echo ""
    echo "To set the token:"
    echo "  export HF_TOKEN=your_huggingface_token"
    echo ""
    echo "Get your token from: https://huggingface.co/settings/tokens"
    exit 1
fi

# Set default checkpoint path
if [ -z "$CHECKPOINT" ]; then
    CHECKPOINT="experiments/$MODE/checkpoint-best"
fi

# Check checkpoint exists
if [ ! -d "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found at $CHECKPOINT"
    echo ""
    echo "Make sure you have trained a model first:"
    echo "  ./scripts/run_train.sh --mode $MODE"
    exit 1
fi

echo "=============================================="
echo "Push Model to Hugging Face Hub"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  Mode: $MODE"
echo "  Checkpoint: $CHECKPOINT"
echo "  Repository: $REPO_NAME"
echo "  Private: $PRIVATE"
echo ""

# Build command
CMD="python -m src.push_to_hf --checkpoint $CHECKPOINT --repo-name $REPO_NAME --mode $MODE"

if [ "$PRIVATE" = true ]; then
    CMD="$CMD --private"
fi

# Check for metrics file
METRICS_PATH="experiments/$MODE/eval_results.json"
if [ -f "$METRICS_PATH" ]; then
    CMD="$CMD --metrics $METRICS_PATH"
    echo "Found metrics at: $METRICS_PATH"
fi

echo ""
echo "Running: $CMD"
echo ""

$CMD

echo ""
echo "=============================================="
echo "Upload complete!"
echo "=============================================="
