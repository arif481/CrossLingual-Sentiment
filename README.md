# Cross-lingual sentiment analysis for low-resource languages

[![CI](https://github.com/arif481/CrossLingual-Sentiment/actions/workflows/ci.yml/badge.svg)](https://github.com/arif481/CrossLingual-Sentiment/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/arif481/CrossLingual-Sentiment/blob/main/notebooks/colab_run.ipynb)

This project tests how well a sentiment classifier trained on English carries over to Bengali. It fine-tunes XLM-RoBERTa base for binary sentiment classification in three setups and compares each with a TF-IDF + logistic regression baseline:

| Setting | Train on | Question it answers |
|---|---|---|
| Zero-shot transfer | English | How far does English-only training carry into Bengali? |
| Low-resource | Bengali | What do about 9,700 Bengali examples give on their own? |
| Combined | English + Bengali | Does adding English data help Bengali, or the other way round? |

Every model is tested on both the English and the Bengali test set. The best epoch is chosen on validation data from the training language only, so the zero-shot model never sees a Bengali label.

## Data

- English: [SST-2](https://huggingface.co/datasets/nyu-mll/glue) from GLUE, movie review sentences. SST-2's test labels are not public, so its validation split (872 sentences) is the English test set, and 5% of the training split (3,368 examples) is held out for validation. That leaves 63,981 training examples.
- Bengali: [SentNoB](https://huggingface.co/datasets/khondoker/SentNoB) (Islam et al., Findings of EMNLP 2021), Bangla social media comments labelled positive, negative or neutral. The paper's own train/val/test files are used with neutral comments removed: 9,681 / 1,213 / 1,225 comments, 53% positive.

`python -m src.data_loader` downloads both from the Hugging Face Hub and writes the splits to `data/processed/`. The data is not stored in this repository; SentNoB is licensed CC BY-ND 4.0.

## Results

Baseline, TF-IDF (unigrams and bigrams, 50,000 features) + logistic regression, run on a CPU with scikit-learn 1.7.2:

| Trained on | English test, macro-F1 | Bengali test, macro-F1 | Bengali test, accuracy |
|---|---|---|---|
| English | 81.4 | 35.4 | 53.3 |
| Bengali | 33.0 | 75.6 | 75.6 |
| English + Bengali | 81.6 | 73.3 | 73.3 |

The English-only baseline labels 1,216 of the 1,225 Bengali comments positive, so its Bengali accuracy is just the share of positive comments. Adding English data to the Bengali training set lowers the Bengali score by 2.3 points: with a 50,000-feature cap, the combined vocabulary keeps only 3,413 Bengali n-grams, against 35,256 when training on Bengali alone.

XLM-RoBERTa results are not in yet. They come from running the Colab notebook below, and every number will be traceable to a `metrics.json` file in `experiments/`.

## Running the experiments

The quickest way is the [Colab notebook](notebooks/colab_run.ipynb) on a free T4 GPU. It prepares the data, runs the baseline and the three XLM-RoBERTa runs, prints the results table and downloads `results.zip`.

To run it locally (Python 3.10 or newer, and a GPU for the transformer runs):

```bash
git clone https://github.com/arif481/CrossLingual-Sentiment.git
cd CrossLingual-Sentiment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m src.data_loader --language all                 # download and split SST-2 and SentNoB
python -m src.baseline --run-all --output experiments/baseline
python -m src.train --mode low_resource                  # or zero_shot, combined; add --seed N
python -m src.report                                     # writes experiments/results.md
```

`python -m src.train --mode combined --max-train-samples 256 --max-eval-samples 128` is a quick check that everything runs.

Hyperparameters are in `configs/config.yaml`: max length 128, batch size 32, learning rate 2e-5 with 10% warmup, weight decay 0.01, up to 4 epochs with early stopping after 2 epochs without a gain in validation macro-F1. The pipeline was tested with transformers 5.20.2, datasets 5.1.0 and torch 2.14.

Each run writes `experiments/<mode>/seed<N>/metrics.json` (test accuracy, macro precision, recall and F1, and the confusion matrix for each language, plus run time and library versions) and `training_history.json`. The trained model goes to `models/<mode>-seed<N>/best/`, which is not committed.

## Published model

The model at [arif481/crosslingual-sentiment-model](https://huggingface.co/arif481/crosslingual-sentiment-model) and the [demo Space](https://huggingface.co/spaces/arif481/crosslingual-demo) date from an early one-epoch run on a handful of demo sentences, before the data pipeline was fixed. Don't rely on them; they will be replaced with the combined model from the full run.

## Layout

```
configs/      hyperparameters and dataset settings
src/          data loading, preprocessing, baseline, training, report, evaluation, HF upload
experiments/  metrics from each run and the results summary
scripts/      shell wrappers for preprocessing, training and upload
app/          Streamlit and Gradio demos
notebooks/    the Colab runner and an analysis notebook
tests/        pytest tests
```

Run the tests with `pytest tests/ -v`.

## Limitations

SST-2 is movie reviews and SentNoB is social media comments, so zero-shot transfer measures a change of domain as well as language. Dropping SentNoB's neutral class makes the task easier than the paper's three-way setup, and the scores are not comparable with it. With 1,225 Bengali test comments, the 95% confidence interval on an accuracy near 75% is about ±2.4 points, so small differences between setups need several seeds before they mean anything; the notebook can repeat runs over more seeds.

## Citation

SentNoB:

```bibtex
@inproceedings{islam2021sentnob,
  title={SentNoB: A Dataset for Analysing Sentiment on Noisy Bangla Texts},
  author={Islam, Khondoker Ittehadul and Kar, Sudipta and Islam, Md Saiful and Amin, Mohammad Ruhul},
  booktitle={Findings of the Association for Computational Linguistics: EMNLP 2021},
  pages={3265--3271},
  year={2021}
}
```

This repository:

```bibtex
@misc{arifuzzaman2026crosslingual,
  author = {Md Arifuzzaman},
  title  = {Cross-Lingual Sentiment Analysis for Low-Resource Languages},
  year   = {2026},
  url    = {https://github.com/arif481/CrossLingual-Sentiment}
}
```

## License

Code: [MIT](LICENSE). Built on [Hugging Face Transformers](https://huggingface.co/docs/transformers) and [XLM-RoBERTa](https://arxiv.org/abs/1911.02116).
