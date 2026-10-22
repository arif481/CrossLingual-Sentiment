Macro-F1 (%) on each test set

| Training data | Model | English (SST-2) | Bengali (SentNoB) |
|---|---|---|---|
| zero_shot | TF-IDF + LR | 81.6 | 35.4 |
| zero_shot | xlm-roberta-base | 92.1 | 67.8 |
| low_resource | TF-IDF + LR | 33.0 | 75.6 |
| low_resource | xlm-roberta-base | 76.9 | 87.4 |
| combined | TF-IDF + LR | 81.2 | 73.1 |
| combined | xlm-roberta-base | 92.0 | 87.9 |

Accuracy (%) on each test set

| Training data | Model | English (SST-2) | Bengali (SentNoB) |
|---|---|---|---|
| zero_shot | TF-IDF + LR | 81.7 | 53.3 |
| zero_shot | xlm-roberta-base | 92.1 | 68.8 |
| low_resource | TF-IDF + LR | 48.9 | 75.6 |
| low_resource | xlm-roberta-base | 77.3 | 87.4 |
| combined | TF-IDF + LR | 81.2 | 73.1 |
| combined | xlm-roberta-base | 92.0 | 87.9 |

Transformer rows average 1 seed(s); ± is the standard deviation across seeds.
