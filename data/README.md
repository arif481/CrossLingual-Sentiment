# Data

Nothing in `processed/` is committed. `python -m src.data_loader --language all` rebuilds it from the Hugging Face Hub:

```
processed/
  en/  train.csv  val.csv  test.csv    SST-2
  bn/  train.csv  val.csv  test.csv    SentNoB, neutral comments removed
```

Each CSV has two columns: `text` and `label` (0 = negative, 1 = positive).

## English: SST-2

[nyu-mll/glue](https://huggingface.co/datasets/nyu-mll/glue), subset `sst2`. The test split has no public labels, so:

| Split here | Taken from | Size |
|---|---|---|
| train | SST-2 train, minus the held-out 5% | 63,981 |
| val | stratified 5% of SST-2 train (seed 42) | 3,368 |
| test | SST-2 validation | 872 |

SST-2's training set contains phrases cut from the same sentences, so the held-out validation set overlaps with training and only serves to pick the best epoch. The test set is separate sentences.

## Bengali: SentNoB

[khondoker/SentNoB](https://huggingface.co/datasets/khondoker/SentNoB), the dataset from Islam et al., "SentNoB: A Dataset for Analysing Sentiment on Noisy Bangla Texts" (Findings of EMNLP 2021). Its labels are 0 = neutral, 1 = positive and 2 = negative. `configs/config.yaml` maps 1 to positive and 2 to negative and the loader drops neutral comments:

| Split here | File | Comments kept | Neutral dropped |
|---|---|---|---|
| train | Train.csv | 9,681 | 2,894 |
| val | Val.csv | 1,213 | 354 |
| test | Test.csv | 1,225 | 361 |

SentNoB is licensed CC BY-ND 4.0, which is why it is downloaded rather than copied into this repository.

## Demo data

`raw/demo/demo_en.csv` and `raw/demo/demo_bn.csv` hold 50 short sentences each, used only by `--use-demo` and the CI smoke test. Scores on them mean nothing.
