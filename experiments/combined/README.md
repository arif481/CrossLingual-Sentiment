# Experiment: Combined Multilingual Training

This directory contains results from the **combined** experiment:
- **Training**: English + Bengali (combined)
- **Evaluation**: Both English and Bengali

## Purpose

This experiment tests whether combining data from multiple languages improves performance, especially on the low-resource language (Bengali).

## Expected Files After Training

```
combined/
├── checkpoint-best/        # Best model checkpoint
├── logs/                   # TensorBoard logs
├── metrics.json            # Training/validation metrics
├── eval_results.json       # Test set evaluation (per-language)
└── error_analysis.json     # Detailed error analysis
```

## Running This Experiment

```bash
# Using the training script
./scripts/run_train.sh --mode combined

# Or directly with Python
python -m src.train --mode combined
```

## Results

Not measured yet. Run the experiment and the scores land in `metrics.json` in this folder. The committed metrics come from a one-epoch run on a small sample that checks the pipeline, not a benchmark.
