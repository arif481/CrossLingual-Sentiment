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
import traceback

# Try multiple models in order of preference
MODELS_TO_TRY = [
    "cardiffnlp/twitter-xlm-roberta-base-sentiment",
    "nlptown/bert-base-multilingual-uncased-sentiment", 
    "lxyuan/distilbert-base-multilingual-cased-sentiments-student",
]

classifier = None
model_name_used = None

print("Loading model...")
for model_name in MODELS_TO_TRY:
    try:
        print(f"Trying: {model_name}")
        classifier = pipeline(
            "sentiment-analysis",
            model=model_name,
            top_k=None
        )
        model_name_used = model_name
        print(f"SUCCESS: Loaded {model_name}")
        break
    except Exception as e:
        print(f"Failed to load {model_name}: {e}")
        continue

if classifier is None:
    print("WARNING: All models failed to load!")

def predict_sentiment(text: str) -> dict:
    """Predict sentiment for input text."""
    if not text or not text.strip():
        return {"Positive": 0.33, "Neutral": 0.34, "Negative": 0.33}
    
    if classifier is None:
        return {"Model Load Error": 1.0}
    
    try:
        results = classifier(text[:512])
        
        # Handle nested list format
        if results and isinstance(results[0], list):
            results = results[0]
        
        output = {}
        for item in results:
            label = item['label'].lower()
            score = item['score']
            
            # Normalize label names
            if 'pos' in label or label == 'positive' or label == '5 stars' or label == '4 stars':
                key = 'Positive'
            elif 'neg' in label or label == 'negative' or label == '1 star' or label == '2 stars':
                key = 'Negative'
            else:
                key = 'Neutral'
            
            # Accumulate scores for same category
            output[key] = output.get(key, 0) + score
        
        # Ensure all keys exist
        for key in ['Positive', 'Neutral', 'Negative']:
            if key not in output:
                output[key] = 0.0
        
        return output
        
    except Exception as e:
        print(f"Prediction error: {e}")
        traceback.print_exc()
        return {"Prediction Error": 1.0}

# Examples
EXAMPLES = [
    ["This product is amazing! I love it so much."],
    ["The service was terrible and I'm very disappointed."],
    ["It's okay, nothing special but works fine."],
    ["এই পণ্যটি অসাধারণ! আমি এটা খুব পছন্দ করি।"],
    ["সেবাটি খুবই খারাপ ছিল এবং আমি খুব হতাশ।"],
]

demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(
        label="Enter Text",
        placeholder="Type text in English or Bengali...",
        lines=3
    ),
    outputs=gr.Label(label="Sentiment", num_top_classes=3),
    title="🌍 Cross-Lingual Sentiment Analysis",
    description=f"""
Analyze sentiment in **English** and **Bengali** text using XLM-RoBERTa.

**Model:** `{model_name_used or 'Loading...'}`

Enter text and get Positive/Neutral/Negative predictions.
""",
    examples=EXAMPLES,
    flagging_mode="never",
    cache_examples=False
)

if __name__ == "__main__":
    demo.launch()
