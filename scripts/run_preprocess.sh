#!/bin/bash
# Run Preprocess Script
# =====================
# Preprocess datasets for cross-lingual sentiment analysis

set -e

# Default values
CONFIG="configs/config.yaml"
DATA_DIR="data/processed"
LANGUAGE="all"
USE_DEMO=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --language)
            LANGUAGE="$2"
            shift 2
            ;;
        --demo)
            USE_DEMO=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --config PATH     Path to config file (default: configs/config.yaml)"
            echo "  --data-dir PATH   Output directory (default: data/processed)"
            echo "  --language LANG   Language to process: en, bn, or all (default: all)"
            echo "  --demo            Use demo data instead of downloading full datasets"
            echo "  --help            Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "Cross-Lingual Sentiment Analysis - Preprocess"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  Config file: $CONFIG"
echo "  Data directory: $DATA_DIR"
echo "  Language: $LANGUAGE"
echo "  Use demo data: $USE_DEMO"
echo ""

# Step 1: Download and prepare data
echo "Step 1: Downloading and preparing data..."

if [ "$USE_DEMO" = true ]; then
    python -m src.data_loader \
        --config "$CONFIG" \
        --language "$LANGUAGE" \
        --processed-dir "$DATA_DIR" \
        --use-demo
else
    python -m src.data_loader \
        --config "$CONFIG" \
        --language "$LANGUAGE" \
        --processed-dir "$DATA_DIR"
fi

# Step 2: Preprocess and tokenize
echo ""
echo "Step 2: Preprocessing and computing statistics..."

python -m src.preprocess \
    --config "$CONFIG" \
    --data-dir "$DATA_DIR" \
    --language "$LANGUAGE" \
    --stats-only

echo ""
echo "=============================================="
echo "Preprocessing complete!"
echo "=============================================="
echo ""
echo "Data saved to: $DATA_DIR"
echo ""
echo "Next steps:"
echo "  1. Run baseline: ./scripts/run_train.sh --baseline"
echo "  2. Train model: ./scripts/run_train.sh --mode zero_shot"
