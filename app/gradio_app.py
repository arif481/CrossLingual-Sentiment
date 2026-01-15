"""
Gradio Demo: Cross-Lingual Sentiment Analysis
==============================================

Interactive demo for cross-lingual sentiment analysis supporting
English and Bengali languages.

Author: Md Arifuzzaman
GitHub: https://github.com/arif481
"""

import gradio as gr
import torch
from transformers import pipeline

# Use a pre-trained multilingual sentiment model that actually works
# cardiffnlp/twitter-xlm-roberta-base-sentiment is trained on multilingual sentiment data
MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

# Load pipeline
print("Loading model...")
try:
    classifier = pipeline(
        "sentiment-analysis",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        top_k=None  # Return all scores
    )
    print(f"Model loaded successfully: {MODEL_NAME}")
except Exception as e:
    print(f"Error loading primary model: {e}")
    classifier = None

def predict_sentiment(text: str) -> dict:
    """
    Predict sentiment for input text.
    
    Args:
        text: Input text (English or Bengali)
        
    Returns:
        Dictionary with sentiment probabilities
    """
    if not text.strip():
        return {"Negative": 0.0, "Neutral": 0.0, "Positive": 0.0}
    
    if classifier is None:
        return {"Error": 1.0}
    
    try:
        # Get predictions
        results = classifier(text[:512])  # Truncate to max length
        
        # Convert to dictionary format expected by Gradio
        if isinstance(results[0], list):
            results = results[0]
        
        output = {}
        for item in results:
            label = item['label'].lower()
            if label in ['negative', 'neg']:
                output['Negative'] = item['score']
            elif label in ['positive', 'pos']:
                output['Positive'] = item['score']
            elif label in ['neutral', 'neu']:
                output['Neutral'] = item['score']
            else:
                output[item['label'].capitalize()] = item['score']
        
        return output
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return {"Error": 1.0}

# Example texts
EXAMPLES = [
    ["This product is amazing! I love it so much."],
    ["The service was terrible and I'm very disappointed."],
    ["It's okay, nothing special but works fine."],
    ["এই পণ্যটি অসাধারণ! আমি এটা খুব পছন্দ করি।"],
    ["সেবাটি খুবই খারাপ ছিল এবং আমি খুব হতাশ।"],
    ["এটা ঠিক আছে, মোটামুটি কাজ চলে।"],
]

# Create Gradio interface
demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(
        label="Enter Text",
        placeholder="Type or paste text in English or Bengali...",
        lines=4
    ),
    outputs=gr.Label(
        label="Sentiment Prediction",
        num_top_classes=3
    ),
    title="🌍 Cross-Lingual Sentiment Analysis",
    description="""
    ## Analyze sentiment in English and Bengali text
    
    This demo uses **XLM-RoBERTa** fine-tuned for multilingual sentiment analysis.
    It supports 8+ languages including English and Bengali.
    
    ### Supported Languages
    - 🇬🇧 **English**
    - 🇧🇩 **Bengali (বাংলা)**
    
    ### How it works
    Enter any text and the model will predict:
    - 😊 **Positive** - Happy, satisfied, enthusiastic
    - 😐 **Neutral** - Objective, factual
    - 😞 **Negative** - Unhappy, disappointed, frustrated
    
    ---
    **GitHub:** [arif481/CrossLingual-Sentiment](https://github.com/arif481/CrossLingual-Sentiment)
    """,
    examples=EXAMPLES,
    flagging_mode="never",
    cache_examples=False
)

if __name__ == "__main__":
    demo.launch()
