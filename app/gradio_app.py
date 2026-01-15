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
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np

# Model configuration
MODEL_NAME = "arif481/crosslingual-sentiment-model"
LABELS = ["Negative", "Neutral", "Positive"]

# Load model and tokenizer
print("Loading model...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    # Fallback to base model for demo
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    model = AutoModelForSequenceClassification.from_pretrained(
        "xlm-roberta-base", 
        num_labels=3
    )
    model.eval()
    print("Using base model as fallback")

def predict_sentiment(text: str) -> dict:
    """
    Predict sentiment for input text.
    
    Args:
        text: Input text (English or Bengali)
        
    Returns:
        Dictionary with sentiment probabilities
    """
    if not text.strip():
        return {label: 0.0 for label in LABELS}
    
    # Tokenize input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0].numpy()
    
    # Create output dictionary
    result = {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}
    
    return result

# Example texts
EXAMPLES = [
    ["This product is amazing! I love it so much."],
    ["The service was terrible and I'm very disappointed."],
    ["It's okay, nothing special but not bad either."],
    ["এই পণ্যটি অসাধারণ! আমি এটা খুব পছন্দ করি।"],  # Bengali: This product is amazing
    ["সেবাটি খুবই খারাপ ছিল এবং আমি খুব হতাশ।"],  # Bengali: The service was terrible
    ["এটা ঠিক আছে, বিশেষ কিছু না তবে খারাপও না।"],  # Bengali: It's okay
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
    
    This model uses **XLM-RoBERTa** fine-tuned for cross-lingual sentiment analysis.
    It can analyze text in both English and Bengali languages.
    
    ### Supported Languages
    - 🇬🇧 **English**
    - 🇧🇩 **Bengali (বাংলা)**
    
    ### How it works
    Enter any text and the model will predict whether the sentiment is:
    - 😊 **Positive** - Happy, satisfied, enthusiastic
    - 😐 **Neutral** - Objective, factual, neither positive nor negative  
    - 😞 **Negative** - Unhappy, disappointed, frustrated
    
    ---
    **Model:** [arif481/crosslingual-sentiment-model](https://huggingface.co/arif481/crosslingual-sentiment-model)  
    **GitHub:** [arif481/CrossLingual-Sentiment](https://github.com/arif481/CrossLingual-Sentiment)
    """,
    examples=EXAMPLES,
    flagging_mode="never"
)

if __name__ == "__main__":
    demo.launch()
