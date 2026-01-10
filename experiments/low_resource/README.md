# Experiment: Low-Resource Fine-tuning

This directory contains results from the **low-resource** experiment:
- **Training**: Bengali only
- **Evaluation**: Bengali only

## Purpose

This experiment establishes a baseline for what can be achieved with limited target language data, without any cross-lingual transfer from high-resource languages.

## Expected Files After Training

```
low_resource/
├── checkpoint-best/        # Best model checkpoint
├── logs/                   # TensorBoard logs
├── metrics.json            # Training/validation metrics
├── eval_results.json       # Test set evaluation
└── error_analysis.json     # Detailed error analysis
```

## Running This Experiment

```bash
# Using the training script
./scripts/run_train.sh --mode low_resource

# Or directly with Python
python -m src.train --mode low_resource
```

## Expected Results

Low-resource training typically shows:
- Decent performance but limited by small dataset
- May overfit without careful regularization
- Provides comparison point for zero-shot and combined approaches

| Metric | Bengali |
|--------|---------|
| Accuracy | ~70-80% |
| F1 Score | ~70-80% |

*Note: Results depend heavily on the amount of Bengali training data available.*
