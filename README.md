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

One run of each setup (seed 42) on a free Colab T4. Macro-F1 (%):

| Trained on | Model | English test | Bengali test |
|---|---|---|---|
| English | TF-IDF + LR | 81.6 | 35.4 |
| English | XLM-R base | 92.1 | 67.8 |
| Bengali | TF-IDF + LR | 33.0 | 75.6 |
| Bengali | XLM-R base | 76.9 | 87.4 |
| English + Bengali | TF-IDF + LR | 81.2 | 73.1 |
| English + Bengali | XLM-R base | 92.0 | 87.9 |

Accuracy is in [experiments/results.md](experiments/results.md); each run's `metrics.json` also has precision, recall and the confusion matrix for both test sets. The baseline is TF-IDF over unigrams and bigrams (50,000 features) with logistic regression.

What the numbers show:

- Zero-shot transfer works but leaves a large gap. Trained only on English movie-review sentences, XLM-R reaches 67.8 on Bengali comments, 32 points above TF-IDF, which labels 1,216 of the 1,225 comments positive. It is still 19.6 points behind training on Bengali. Its errors mostly go one way: it gets 93% of negative comments right but calls 342 of the 654 positive ones (52%) negative.
- Bengali data does most of the work. 9,681 Bengali comments give 87.4; adding 63,981 English sentences gives 87.9. The 0.5-point gap is well inside the 95% confidence interval of about ±1.9 points for 1,225 test comments, so on one seed the combined model cannot be called better. English barely changes either (92.1 against 92.0).
- Transfer also runs the other way: the Bengali-only model scores 76.9 on English without seeing an English example.
- For TF-IDF, adding English costs 2.5 points on Bengali. Under the 50,000-feature cap the combined vocabulary keeps only 3,413 Bengali n-grams, against 35,256 when training on Bengali alone.

Training took 4.8 minutes (Bengali only), 18.5 minutes (English only) and 22 minutes (combined), with torch 2.11, transformers 5.20.2 and datasets 5.1.0.

## Running the experiments

The quickest way is the [Colab notebook](notebooks/colab_run.ipynb) on a free T4 GPU, about 50 minutes end to end. It prepares the data, runs the baseline and the three XLM-RoBERTa runs, prints the results table and downloads `results.zip`.

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
python -m src.learning_curve                             # needs models/zero_shot-seed42/best from the zero_shot run
```

`python -m src.train --mode combined --max-train-samples 256 --max-eval-samples 128` is a quick check that everything runs.

Hyperparameters are in `configs/config.yaml`: max length 128, batch size 32, learning rate 2e-5 with 10% warmup, weight decay 0.01, up to 4 epochs with early stopping after 2 epochs without a gain in validation macro-F1. The notebook pins transformers 5.20.2, datasets 5.1.0 and accelerate 1.15.0.

Each run writes `experiments/<mode>/seed<N>/metrics.json` (test accuracy, macro precision, recall and F1, and the confusion matrix for each language, plus run time and library versions) and `training_history.json`. The trained model goes to `models/<mode>-seed<N>/best/`, which is not committed.

## Published model

The model at [arif481/crosslingual-sentiment-model](https://huggingface.co/arif481/crosslingual-sentiment-model) and the [demo Space](https://huggingface.co/spaces/arif481/crosslingual-demo) date from an early one-epoch run on a handful of demo sentences, before the data pipeline was fixed. Don't rely on them; they will be replaced with the combined model from the full run.

## Layout

```
configs/      hyperparameters and dataset settings
src/          data loading, preprocessing, baseline, training, learning curve, report, evaluation, HF upload
experiments/  metrics from each run and the results summary
scripts/      shell wrappers for preprocessing, training and upload
app/          Streamlit and Gradio demos
notebooks/    the Colab runner and an analysis notebook
tests/        pytest tests
```

Run the tests with `pytest tests/ -v`.

## Limitations

SST-2 is movie reviews and SentNoB is social media comments, so zero-shot transfer measures a change of domain as well as language. Dropping SentNoB's neutral class makes the task easier than the paper's three-way setup, and the scores are not comparable with it. All results are from a single seed, and with 1,225 Bengali test comments differences of a point or two are noise; setting `SEEDS = [42, 43, 44]` in the notebook repeats every run and reports the mean and standard deviation. The only multilingual encoder tested is XLM-R base; a Bangla-specific model such as BanglaBERT is the obvious comparison still missing.

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
