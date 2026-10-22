---
title: Cross-lingual Sentiment
emoji: 🌖
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 6.3.0
app_file: app.py
pinned: false
license: mit
models:
  - arif481/crosslingual-sentiment-model
---

Demo of [arif481/crosslingual-sentiment-model](https://huggingface.co/arif481/crosslingual-sentiment-model): XLM-RoBERTa base fine-tuned to label English and Bengali text as negative or positive. Test macro-F1 is 92.0 on SST-2 (English) and 87.9 on SentNoB (Bengali).

The code, data pipeline and experiments are in [arif481/CrossLingual-Sentiment](https://github.com/arif481/CrossLingual-Sentiment). This Space is uploaded from `app/` there by `python -m src.push_to_hf --space crosslingual-demo`.
