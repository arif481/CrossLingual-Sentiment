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

## Expected Results

Combined training typically shows:
- Strong performance on both languages
- Bengali performance often improves over low-resource-only
- May slightly decrease English performance (negative transfer)

| Metric | English | Bengali |
|--------|---------|---------|
| Accuracy | ~88% | ~78-85% |
| F1 Score | ~88% | ~75-82% |

*Note: Results depend on language balance and dataset sizes.*

## Key Findings

1. **Positive Transfer**: Bengali benefits from English training data
2. **Balanced Training**: Important to balance language representation
3. **Best Overall**: Often the best choice for multilingual deployment
