# Cross-lingual sentiment analysis for low-resource languages

[![CI](https://github.com/arif481/CrossLingual-Sentiment/actions/workflows/ci.yml/badge.svg)](https://github.com/arif481/CrossLingual-Sentiment/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Model on Hugging Face](https://img.shields.io/badge/Hugging%20Face-model-orange)](https://huggingface.co/arif481/crosslingual-sentiment-model)

This project tests how well a sentiment classifier trained on English carries over to Bengali. It fine-tunes XLM-RoBERTa for binary sentiment classification and compares three training setups:

| Setting | Train on | Evaluate on | Question it answers |
|---|---|---|---|
| Zero-shot transfer | English (SST-2) | Bengali | How far does English-only training carry? |
| Low-resource | Bengali | Bengali | What do a few Bengali examples alone give you? |
| Combined | English + Bengali | both | Does mixing the two beat either one? |

A TF-IDF + logistic regression baseline sets the floor, since word features can't transfer between languages.

## Status

The data pipeline, baseline, all three training modes, evaluation, the Hugging Face upload and a Streamlit/Gradio demo are in place, with tests and CI. The metrics committed in `experiments/combined/metrics.json` come from a one-epoch demo run on a small sample, used to check the pipeline end to end; they are not a benchmark. [paper.md](paper.md) is a working draft.

## Setup

Python 3.10 or newer.

```bash
git clone https://github.com/arif481/CrossLingual-Sentiment.git
cd CrossLingual-Sentiment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Running the experiments

```bash
./scripts/run_preprocess.sh --demo            # small sample for a quick check
./scripts/run_preprocess.sh --language all    # full English and Bengali datasets

./scripts/run_train.sh --baseline             # TF-IDF + logistic regression
./scripts/run_train.sh --mode zero_shot       # or low_resource, combined
./scripts/run_train.sh --mode combined --demo # one epoch, for testing

python -m src.evaluate --checkpoint experiments/combined/checkpoint-best --mode combined
```

Hyperparameters live in `configs/config.yaml` (default: `xlm-roberta-base`, max length 128, 5 epochs, batch size 16, learning rate 2e-5, early stopping after 3 epochs without improvement). Seeds are fixed for reproducibility.

To upload a trained model, set `HF_TOKEN` and run `./scripts/push_model.sh --mode combined`.

## Using the published model

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis", model="arif481/crosslingual-sentiment-model")
classifier("This movie is absolutely fantastic!")
classifier("এই সিনেমাটি অসাধারণ ছিল!")
```

There is also a [demo Space](https://huggingface.co/spaces/arif481/crosslingual-demo), and `streamlit run app/streamlit_app.py` runs the same demo locally.

## Layout

```
configs/      hyperparameters
src/          data loading, preprocessing, baseline, model, training, evaluation, HF upload
experiments/  outputs for zero_shot, low_resource and combined
scripts/      preprocessing, training and upload entry points
app/          Streamlit and Gradio demos
notebooks/    analysis and a Colab runner
tests/        pytest tests for data loading and preprocessing
```

Run the tests with `pytest tests/ -v`.

## Limitations

The model can pick up biases from its training data, and sentiment in Bengali often depends on cultural context that SST-2 doesn't cover. It isn't meant for production use without further validation.

## Citation

```bibtex
@misc{arifuzzaman2026crosslingual,
  author = {Md Arifuzzaman},
  title  = {Cross-Lingual Sentiment Analysis for Low-Resource Languages},
  year   = {2026},
  url    = {https://github.com/arif481/CrossLingual-Sentiment}
}
```

## License

[MIT](LICENSE). Built on [Hugging Face Transformers](https://huggingface.co/docs/transformers) and [XLM-RoBERTa](https://arxiv.org/abs/1911.02116).
