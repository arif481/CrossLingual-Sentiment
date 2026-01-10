# Experiment: Zero-Shot Cross-Lingual Transfer

This directory contains results from the **zero-shot** experiment:
- **Training**: English only
- **Evaluation**: Bengali only

## Expected Files After Training

```
zero_shot/
├── checkpoint-best/        # Best model checkpoint
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── ...
├── checkpoint-*/           # Intermediate checkpoints
├── logs/                   # TensorBoard logs
├── metrics.json            # Training/validation metrics
├── training_history.json   # Full training log
├── eval_results.json       # Test set evaluation
├── eval_report.html        # HTML evaluation report
└── error_analysis.json     # Detailed error analysis
```

## Running This Experiment

```bash
# Using the training script
./scripts/run_train.sh --mode zero_shot

# Or directly with Python
python -m src.train --mode zero_shot
```

## Evaluating

```bash
python -m src.evaluate \
    --checkpoint experiments/zero_shot/checkpoint-best \
    --mode zero_shot
```

## Expected Results

Zero-shot cross-lingual transfer typically shows:
- High accuracy on English (source language)
- Lower but meaningful accuracy on Bengali (target language)
- Performance gap indicates room for improvement with target language data

| Metric | English | Bengali |
|--------|---------|---------|
| Accuracy | ~90% | ~65-75% |
| F1 Score | ~90% | ~60-70% |

*Note: Actual results depend on dataset size and model configuration.*
