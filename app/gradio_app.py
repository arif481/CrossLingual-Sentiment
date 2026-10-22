"""
Gradio Demo: Cross-Lingual Sentiment Analysis
==============================================

Interactive demo for cross-lingual sentiment analysis supporting
English and Bengali languages. Deployed as the Hugging Face Space
arif481/crosslingual-demo by `python -m src.push_to_hf --space crosslingual-demo`.

Author: Md Arifuzzaman
GitHub: https://github.com/arif481
"""

import gradio as gr
from transformers import pipeline

MODEL_NAME = "arif481/crosslingual-sentiment-model"

print("Loading model...")
try:
    classifier = pipeline(
        "text-classification",
        model=MODEL_NAME,
        top_k=None
    )
    print(f"Model loaded: {MODEL_NAME}")
except Exception as e:
    print(f"Error: {e}")
    classifier = None

def predict_sentiment(text: str) -> dict:
    """Predict sentiment for input text."""
    if not text or not text.strip():
        return {}

    if classifier is None:
        return {"Error - Model not loaded": 1.0}

    try:
        results = classifier(text, truncation=True)

        if results and isinstance(results[0], list):
            results = results[0]

        # The model's labels are "negative" and "positive"
        return {item["label"].capitalize(): item["score"] for item in results}

    except Exception as e:
        print(f"Error: {e}")
        return {"Prediction Error": 1.0}

# Examples
EXAMPLES = [
    ["This product is amazing! I love it so much."],
    ["The service was terrible and I'm very disappointed."],
    ["I had a wonderful experience, highly recommended!"],
    ["এই পণ্যটি অসাধারণ! আমি এটা খুব পছন্দ করি।"],
    ["সেবাটি খুবই খারাপ ছিল এবং আমি খুব হতাশ।"],
    ["চমৎকার অভিজ্ঞতা, সবাইকে সুপারিশ করব!"],
]

demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(
        label="Enter Text",
        placeholder="Type text in English or Bengali...",
        lines=3
    ),
    outputs=gr.Label(label="Sentiment", num_top_classes=2),
    title="Cross-lingual sentiment analysis",
    description="""
XLM-RoBERTa base fine-tuned by Md Arifuzzaman on English movie-review sentences (SST-2) and Bengali social media comments (SentNoB).
Test macro-F1: 92.0 on English, 87.9 on Bengali. There are only two labels, so neutral text still gets one of them.

**Model:** [arif481/crosslingual-sentiment-model](https://huggingface.co/arif481/crosslingual-sentiment-model) ·
**Code and experiments:** [arif481/CrossLingual-Sentiment](https://github.com/arif481/CrossLingual-Sentiment)
""",
    examples=EXAMPLES,
    flagging_mode="never",
    cache_examples=False
)

if __name__ == "__main__":
    demo.launch()
