# Cross-Lingual Sentiment Analysis for Low-Resource Languages

[![CI](https://github.com/arif481/crosslingual-sentiment/actions/workflows/ci.yml/badge.svg)](https://github.com/arif481/crosslingual-sentiment/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-orange)](https://huggingface.co/arif481/crosslingual-sentiment-model)

A research-ready framework for cross-lingual sentiment analysis using multilingual transformers (XLM-RoBERTa). Supports English (high-resource) and Bengali (low-resource) with three experimental configurations: zero-shot transfer, low-resource fine-tuning, and combined multilingual training.

## 🌟 Features

- **Binary Sentiment Classification**: Positive/Negative classification
- **Cross-Lingual Transfer**: Train on English, evaluate on Bengali
- **Multiple Experiments**: Zero-shot, low-resource, and combined modes
- **Reproducible Research**: Seed control, comprehensive logging, result tracking
- **Easy Deployment**: Push to Hugging Face Hub, Streamlit demo, GitHub Pages
- **CI/CD Ready**: GitHub Actions for testing and deployment

## 📁 Repository Structure

```
crosslingual-sentiment/
├── configs/
│   └── config.yaml           # Hyperparameters and settings
├── data/
│   ├── raw/                  # Downloaded datasets
│   ├── processed/            # Tokenized splits
│   └── README.md             # Dataset documentation
├── src/
│   ├── data_loader.py        # Data download and preparation
│   ├── preprocess.py         # Text cleaning and tokenization
│   ├── baseline.py           # TF-IDF + LogisticRegression
│   ├── model.py              # Model building utilities
│   ├── train.py              # Training with HF Trainer
│   ├── evaluate.py           # Evaluation and metrics
│   ├── push_to_hf.py         # Hugging Face Hub upload
│   └── utils.py              # Helper functions
├── experiments/
│   ├── zero_shot/            # English → Bengali transfer
│   ├── low_resource/         # Bengali only
│   └── combined/             # Multilingual training
├── notebooks/
│   ├── analysis.ipynb        # Results visualization
│   └── colab_run.ipynb       # Google Colab demo
├── app/
│   └── streamlit_app.py      # Web demo application
├── tests/
│   ├── test_data.py          # Data module tests
│   └── test_preprocess.py    # Preprocessing tests
├── scripts/
│   ├── run_preprocess.sh     # Data preparation
│   ├── run_train.sh          # Model training
│   └── push_model.sh         # HF Hub upload
├── docs/
│   └── index.html            # GitHub Pages site
├── .github/workflows/
│   ├── ci.yml                # Continuous integration
│   ├── push-model-to-hf.yml  # Model deployment
│   └── docs-deploy.yml       # Documentation deployment
├── requirements.txt
├── README.md
├── LICENSE
└── paper.md                  # Draft research paper
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/arif481/crosslingual-sentiment.git
cd crosslingual-sentiment

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Using the Pre-trained Model

```python
from transformers import pipeline

# Load model from Hugging Face Hub
classifier = pipeline(
    "sentiment-analysis",
    model="arif481/crosslingual-sentiment-model"
)

# English
result = classifier("This movie is absolutely fantastic!")
print(result)  # [{'label': 'positive', 'score': 0.98}]

# Bengali
result = classifier("এই সিনেমাটি অসাধারণ ছিল!")
print(result)  # [{'label': 'positive', 'score': 0.95}]
```

### Training Your Own Model

#### Step 1: Prepare Data

```bash
# Use demo data (small subset for quick testing)
./scripts/run_preprocess.sh --demo

# Or download full datasets
./scripts/run_preprocess.sh --language all
```

#### Step 2: Run Baseline

```bash
# TF-IDF + Logistic Regression baseline
./scripts/run_train.sh --baseline
```

#### Step 3: Train Transformer Model

```bash
# Zero-shot: Train on English, test on Bengali
./scripts/run_train.sh --mode zero_shot

# Low-resource: Train and test on Bengali only
./scripts/run_train.sh --mode low_resource

# Combined: Train on both languages
./scripts/run_train.sh --mode combined

# Quick demo mode (1 epoch)
./scripts/run_train.sh --mode combined --demo
```

#### Step 4: Evaluate

```bash
python -m src.evaluate \
    --checkpoint experiments/combined/checkpoint-best \
    --mode combined
```

#### Step 5: Push to Hugging Face Hub

```bash
# Set your token
export HF_TOKEN=your_huggingface_token  # Get from https://huggingface.co/settings/tokens

# Push model
./scripts/push_model.sh --mode combined
```

## 📊 Expected Results

| Experiment | Train | Eval | Accuracy | Macro-F1 |
|------------|-------|------|----------|----------|
| Zero-Shot | EN | BN | ~72% | ~70% |
| Low-Resource | BN | BN | ~78% | ~77% |
| Combined | EN+BN | EN+BN | ~82% | ~81% |
| Baseline (TF-IDF) | EN | BN | ~55% | ~52% |

*Results depend on dataset size and hyperparameters.*

## 🎯 Experiment Modes

### Zero-Shot Transfer
Train exclusively on English data and evaluate on Bengali without any Bengali training examples. Tests the cross-lingual capabilities of XLM-RoBERTa.

### Low-Resource Fine-tuning
Train only on limited Bengali data. Establishes what's achievable with target language data alone.

### Combined Multilingual
Train on concatenated English and Bengali data. Typically achieves best overall performance through positive transfer.

## 🌐 Demo & Deployment

### Run Streamlit Demo Locally

```bash
streamlit run app/streamlit_app.py
```

### Deploy to Hugging Face Spaces

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space)
2. Select "Streamlit" as the SDK
3. Copy `app/streamlit_app.py` to the Space
4. Add a `requirements.txt`:
   ```
   transformers
   torch
   streamlit
   ```
5. Update `HF_MODEL_ID` in the app to your model

### Enable GitHub Pages

1. Go to repository Settings → Pages
2. Source: Deploy from branch `main`, folder `/docs`
3. Your site will be at `https://arif481.github.io/crosslingual-sentiment/`

## 🔧 Configuration

Edit `configs/config.yaml` to customize:

```yaml
model:
  default_model: "xlm-roberta-base"
  max_length: 128
  dropout: 0.1

training:
  epochs: 5
  batch_size: 16
  learning_rate: 2.0e-5
  early_stopping_patience: 3
```

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

## 📝 Research Paper

See [paper.md](paper.md) for a draft research paper including:
- Abstract and Introduction
- Related Work
- Methodology
- Experiments and Results
- Analysis and Discussion
- Ethical Considerations

## 🔒 Security & Ethics

### Security
- Never commit API tokens or credentials
- Use environment variables or GitHub Secrets
- See [SECURITY.md](SECURITY.md) for reporting vulnerabilities

### Ethical Considerations
- Models may reflect biases in training data
- Bengali sentiment may include cultural context
- Not intended for production deployment without further validation
- See ethics section in paper.md

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## 📚 Citation

If you use this work, please cite:

```bibtex
@misc{crosslingual-sentiment,
  author = {Md Arifuzzaman},
  title = {Cross-Lingual Sentiment Analysis for Low-Resource Languages},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/arif481/crosslingual-sentiment}
}
```

## 🙏 Acknowledgments

- [Hugging Face](https://huggingface.co/) for Transformers and Hub
- [XLM-RoBERTa](https://arxiv.org/abs/1911.02116) for the multilingual model
- SST-2 and Bengali sentiment dataset contributors

---

**Links:**
- [GitHub Repository](https://github.com/arif481/crosslingual-sentiment)
- [Hugging Face Model](https://huggingface.co/arif481/crosslingual-sentiment-model)
- [Live Demo](https://huggingface.co/spaces/arif481/crosslingual-demo)
- [Project Page](https://arif481.github.io/crosslingual-sentiment/)
