# Data Directory

This directory contains raw and processed datasets for cross-lingual sentiment analysis.

## Directory Structure

```
data/
├── raw/                    # Downloaded raw datasets
│   ├── en/                 # English datasets
│   ├── bn/                 # Bengali datasets
│   └── demo/               # Small demo subset
├── processed/              # Tokenized and split datasets
│   ├── en/
│   │   ├── train.csv
│   │   ├── val.csv
│   │   └── test.csv
│   └── bn/
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
└── README.md
```

## Dataset Sources

### English Dataset (SST-2)
- **Source**: Stanford Sentiment Treebank (via GLUE benchmark)
- **Hugging Face ID**: `glue` (subset: `sst2`)
- **Direct URL**: https://huggingface.co/datasets/glue/viewer/sst2
- **Size**: ~67K training examples, ~872 validation examples
- **Labels**: 0 (negative), 1 (positive)
- **Download**: Automatically downloaded via `datasets` library

```python
from datasets import load_dataset
dataset = load_dataset("glue", "sst2")
```

### Bengali Dataset
- **Primary Source**: Bengali Sentiment Dataset on Hugging Face
- **Hugging Face ID**: `sepidmnorozy/Bengali_sentiment`
- **Direct URL**: https://huggingface.co/datasets/sepidmnorozy/Bengali_sentiment
- **Alternative**: Kaggle Bengali Sentiment datasets

```python
from datasets import load_dataset
dataset = load_dataset("sepidmnorozy/Bengali_sentiment")
```

### Alternative Bengali Sources (Kaggle)

If the Hugging Face dataset is unavailable:

1. **Bengali Sentiment Analysis Dataset**
   - URL: https://www.kaggle.com/datasets/cryptexcode/bangla-sentiment-analysis
   
2. **Setting up Kaggle API**:
   ```bash
   # Install kaggle CLI
   pip install kaggle
   
   # Create API token at https://www.kaggle.com/account
   # Download kaggle.json and place it at:
   mkdir -p ~/.kaggle
   mv kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   
   # Download dataset
   kaggle datasets download -d cryptexcode/bangla-sentiment-analysis -p data/raw/bn/
   unzip data/raw/bn/bangla-sentiment-analysis.zip -d data/raw/bn/
   ```

## Label Mapping

All datasets are standardized to binary labels:
- `0`: Negative sentiment
- `1`: Positive sentiment

## Data Processing Pipeline

1. **Download**: Run `python -m src.data_loader`
2. **Preprocess**: Run `python -m src.preprocess`
3. **Tokenize**: Cached tokenized datasets stored in `processed/`

## Demo Data

A small subset (~100 samples per language) is included in `data/raw/demo/` for quick testing:
- `demo_en.csv`: English samples
- `demo_bn.csv`: Bengali samples

## Privacy and Ethics

- Datasets contain publicly available text from reviews/social media
- No personally identifiable information (PII) is stored
- Bengali data may contain cultural/regional sentiments - use responsibly
- See main README for ethical considerations
