"""
Gradio Demo: Cross-Lingual Sentiment Analysis
==============================================

Interactive demo for cross-lingual sentiment analysis supporting
English and Bengali languages.

Author: Md Arifuzzaman
GitHub: https://github.com/arif481
"""

import gradio as gr
from transformers import pipeline

# Use our trained model
MODEL_NAME = "arif481/crosslingual-sentiment-model"

print("Loading model...")
try:
    classifier = pipeline(
        "sentiment-analysis",
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
        return {"Positive": 0.5, "Negative": 0.5}
    
    if classifier is None:
        return {"Error - Model not loaded": 1.0}
    
    try:
        results = classifier(text[:512])
        
        if results and isinstance(results[0], list):
            results = results[0]
        
        output = {}
        for item in results:
            label = item['label']
            # Map LABEL_0 -> Negative, LABEL_1 -> Positive
            if label == 'LABEL_0':
                output['Negative'] = item['score']
            elif label == 'LABEL_1':
                output['Positive'] = item['score']
            else:
                output[label] = item['score']
        
        return output
        
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
    title="🌍 Cross-Lingual Sentiment Analysis",
    description="""
## Analyze sentiment in English and Bengali text

This model is **XLM-RoBERTa** fine-tuned by **Md Arifuzzaman** on multilingual sentiment data.

**Training:** 4,140 samples (English + Bengali), 1 epoch  
**Validation Accuracy:** 100%

### Supported Languages
- 🇬🇧 **English**
- 🇧🇩 **Bengali (বাংলা)**

---
**Model:** [arif481/crosslingual-sentiment-model](https://huggingface.co/arif481/crosslingual-sentiment-model)  
**GitHub:** [arif481/CrossLingual-Sentiment](https://github.com/arif481/CrossLingual-Sentiment)
""",
    examples=EXAMPLES,
    flagging_mode="never",
    cache_examples=False
)

if __name__ == "__main__":
    demo.launch()
