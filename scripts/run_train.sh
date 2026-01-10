#!/bin/bash
# Run Training Script
# ===================
# Train cross-lingual sentiment models

set -e

# Default values
CONFIG="configs/config.yaml"
DATA_DIR="data/processed"
MODE=""
OUTPUT_DIR=""
EPOCHS=""
BATCH_SIZE=""
DEMO=false
BASELINE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --demo)
            DEMO=true
            shift
            ;;
        --baseline)
            BASELINE=true
            shift
            ;;
        --all)
            # Run all experiments
            echo "Running all experiments..."
            $0 --mode zero_shot --config "$CONFIG" --data-dir "$DATA_DIR"
            $0 --mode low_resource --config "$CONFIG" --data-dir "$DATA_DIR"
            $0 --mode combined --config "$CONFIG" --data-dir "$DATA_DIR"
            exit 0
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --mode MODE       Training mode: zero_shot, low_resource, combined"
            echo "  --config PATH     Path to config file (default: configs/config.yaml)"
            echo "  --data-dir PATH   Data directory (default: data/processed)"
            echo "  --output-dir PATH Output directory (overrides config)"
            echo "  --epochs N        Number of epochs (overrides config)"
            echo "  --batch-size N    Batch size (overrides config)"
            echo "  --demo            Run in demo mode (1 epoch, quick)"
            echo "  --baseline        Run TF-IDF baseline instead of transformer"
            echo "  --all             Run all three experiments"
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
echo "Cross-Lingual Sentiment Analysis - Training"
echo "=============================================="
echo ""

# Run baseline experiments
if [ "$BASELINE" = true ]; then
    echo "Running TF-IDF + Logistic Regression baseline..."
    echo ""
    
    python -m src.baseline \
        --config "$CONFIG" \
        --data "$DATA_DIR" \
        --output "experiments/baseline" \
        --run-all
    
    echo ""
    echo "Baseline training complete!"
    echo "Results saved to: experiments/baseline/"
    exit 0
fi

# Validate mode for transformer training
if [ -z "$MODE" ]; then
    echo "Error: --mode is required for transformer training"
    echo "Use --mode zero_shot|low_resource|combined"
    echo "Or use --baseline for TF-IDF baseline"
    exit 1
fi

echo "Configuration:"
echo "  Mode: $MODE"
echo "  Config file: $CONFIG"
echo "  Data directory: $DATA_DIR"
if [ -n "$OUTPUT_DIR" ]; then
    echo "  Output directory: $OUTPUT_DIR"
fi
if [ -n "$EPOCHS" ]; then
    echo "  Epochs: $EPOCHS"
fi
if [ -n "$BATCH_SIZE" ]; then
    echo "  Batch size: $BATCH_SIZE"
fi
echo "  Demo mode: $DEMO"
echo ""

# Build command
CMD="python -m src.train --mode $MODE --config $CONFIG --data-dir $DATA_DIR"

if [ -n "$OUTPUT_DIR" ]; then
    CMD="$CMD --output-dir $OUTPUT_DIR"
fi

if [ -n "$EPOCHS" ]; then
    CMD="$CMD --epochs $EPOCHS"
fi

if [ -n "$BATCH_SIZE" ]; then
    CMD="$CMD --batch-size $BATCH_SIZE"
fi

if [ "$DEMO" = true ]; then
    CMD="$CMD --demo"
fi

# Run training
echo "Running: $CMD"
echo ""

$CMD

echo ""
echo "=============================================="
echo "Training complete!"
echo "=============================================="
echo ""
echo "Model saved to: experiments/$MODE/checkpoint-best"
echo ""
echo "Next steps:"
echo "  1. Evaluate: python -m src.evaluate --checkpoint experiments/$MODE/checkpoint-best --mode $MODE"
echo "  2. Push to HF: ./scripts/push_model.sh --mode $MODE"
