"""
Streamlit Demo App for Cross-Lingual Sentiment Analysis
========================================================

A web interface for sentiment classification in English and Bengali.
"""

import os
import logging
from typing import Optional, Tuple, List, Dict
import streamlit as st
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Cross-Lingual Sentiment Analysis",
    page_icon="🌐",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Constants
DEFAULT_MODEL = "xlm-roberta-base"  # Fallback
HF_MODEL_ID = os.environ.get(
    "HF_MODEL_ID", 
    "YOUR_USERNAME/crosslingual-sentiment-model"  # Replace with your model
)
LOCAL_MODEL_PATH = "models/demo-checkpoint"


@st.cache_resource
def load_model(model_id: str):
    """
    Load the sentiment analysis model and tokenizer.
    
    Args:
        model_id: Hugging Face model ID or local path
        
    Returns:
        Tuple of (pipeline, model_name)
    """
    from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
    
    # Try loading from HF Hub first
    try:
        logger.info(f"Loading model from HF Hub: {model_id}")
        classifier = pipeline(
            "sentiment-analysis",
            model=model_id,
            device=0 if torch.cuda.is_available() else -1
        )
        return classifier, model_id
    except Exception as e:
        logger.warning(f"Could not load from HF Hub: {e}")
    
    # Try local model
    if os.path.exists(LOCAL_MODEL_PATH):
        try:
            logger.info(f"Loading local model from: {LOCAL_MODEL_PATH}")
            classifier = pipeline(
                "sentiment-analysis",
                model=LOCAL_MODEL_PATH,
                device=0 if torch.cuda.is_available() else -1
            )
            return classifier, LOCAL_MODEL_PATH
        except Exception as e:
            logger.warning(f"Could not load local model: {e}")
    
    # Fallback to base model (for demo)
    logger.info(f"Loading fallback model: {DEFAULT_MODEL}")
    st.warning(
        "⚠️ Using base XLM-RoBERTa model (not fine-tuned). "
        "For best results, deploy your trained model."
    )
    
    # Create a simple classifier with base model
    model = AutoModelForSequenceClassification.from_pretrained(
        DEFAULT_MODEL,
        num_labels=2,
        id2label={0: "negative", 1: "positive"},
        label2id={"negative": 0, "positive": 1}
    )
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    
    classifier = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1
    )
    
    return classifier, DEFAULT_MODEL


def get_sentiment_emoji(label: str, score: float) -> str:
    """Get emoji based on sentiment and confidence."""
    if label.lower() == "positive":
        if score > 0.9:
            return "😄"
        elif score > 0.7:
            return "🙂"
        else:
            return "😐"
    else:
        if score > 0.9:
            return "😢"
        elif score > 0.7:
            return "😕"
        else:
            return "😐"


def highlight_text(text: str, label: str) -> str:
    """Create highlighted text HTML."""
    color = "#90EE90" if label.lower() == "positive" else "#FFB6C1"
    return f'<span style="background-color: {color}; padding: 2px 5px; border-radius: 3px;">{text}</span>'


def predict_sentiment(classifier, text: str) -> Dict:
    """
    Predict sentiment for input text.
    
    Args:
        classifier: Hugging Face pipeline
        text: Input text
        
    Returns:
        Dictionary with label, score, and formatted results
    """
    if not text.strip():
        return None
    
    result = classifier(text)[0]
    
    # Normalize label names
    label = result["label"]
    if label in ["LABEL_0", "NEGATIVE"]:
        label = "negative"
    elif label in ["LABEL_1", "POSITIVE"]:
        label = "positive"
    
    return {
        "label": label,
        "score": result["score"],
        "emoji": get_sentiment_emoji(label, result["score"])
    }


def main():
    """Main Streamlit app."""
    
    # Header
    st.title("🌐 Cross-Lingual Sentiment Analysis")
    st.markdown(
        """
        Analyze sentiment in **English** and **Bengali** using multilingual transformers.
        
        This model uses XLM-RoBERTa for cross-lingual sentiment classification.
        """
    )
    
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    
    language = st.sidebar.selectbox(
        "Language",
        options=["English", "Bengali", "Auto-detect"],
        index=0,
        help="Select the language of your input text"
    )
    
    st.sidebar.markdown("---")
    
    st.sidebar.markdown("### About")
    st.sidebar.markdown(
        """
        This demo showcases cross-lingual sentiment analysis using 
        multilingual transformer models.
        
        **Features:**
        - Binary classification (positive/negative)
        - Zero-shot cross-lingual transfer
        - Support for English and Bengali
        
        [📖 View Code](https://github.com/YOUR_USERNAME/crosslingual-sentiment)
        """
    )
    
    # Load model
    with st.spinner("Loading model..."):
        classifier, model_name = load_model(HF_MODEL_ID)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Model:** `{model_name}`")
    
    # Main content
    st.markdown("---")
    
    # Text input
    st.subheader("📝 Enter Text")
    
    # Example texts
    examples = {
        "English": [
            "This movie is absolutely fantastic! I loved every moment.",
            "Terrible experience. Complete waste of time and money.",
            "The product quality is decent but delivery was slow."
        ],
        "Bengali": [
            "এই সিনেমাটি অসাধারণ ছিল! প্রতিটি মুহূর্ত উপভোগ করেছি।",
            "ভয়ানক অভিজ্ঞতা। সময় এবং টাকার সম্পূর্ণ অপচয়।",
            "পণ্যের গুণমান ঠিক আছে কিন্তু ডেলিভারি ধীর ছিল।"
        ]
    }
    
    # Example selector
    if language != "Auto-detect":
        selected_examples = examples.get(language, examples["English"])
        example_text = st.selectbox(
            "Try an example:",
            options=[""] + selected_examples,
            format_func=lambda x: x[:50] + "..." if len(x) > 50 else x if x else "Select an example..."
        )
    else:
        example_text = ""
    
    # Text area
    user_input = st.text_area(
        "Your text:",
        value=example_text,
        height=100,
        placeholder="Enter text to analyze sentiment..."
    )
    
    # Analyze button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        analyze_button = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)
    
    # Results
    if analyze_button and user_input:
        st.markdown("---")
        st.subheader("📊 Results")
        
        with st.spinner("Analyzing..."):
            result = predict_sentiment(classifier, user_input)
        
        if result:
            # Display result
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    label="Sentiment",
                    value=f"{result['emoji']} {result['label'].upper()}"
                )
            
            with col2:
                st.metric(
                    label="Confidence",
                    value=f"{result['score']:.1%}"
                )
            
            # Progress bar for confidence
            sentiment_color = "green" if result["label"] == "positive" else "red"
            st.progress(result["score"])
            
            # Interpretation
            st.markdown("### 💡 Interpretation")
            
            if result["score"] > 0.9:
                confidence_text = "very high confidence"
            elif result["score"] > 0.7:
                confidence_text = "high confidence"
            elif result["score"] > 0.5:
                confidence_text = "moderate confidence"
            else:
                confidence_text = "low confidence"
            
            st.markdown(
                f"The model predicts this text expresses **{result['label']}** "
                f"sentiment with **{confidence_text}** ({result['score']:.1%})."
            )
    
    elif analyze_button and not user_input:
        st.warning("Please enter some text to analyze.")
    
    # Batch analysis section
    st.markdown("---")
    with st.expander("📋 Batch Analysis"):
        st.markdown("Analyze multiple texts at once (one per line):")
        
        batch_input = st.text_area(
            "Batch input:",
            height=150,
            placeholder="Enter multiple texts, one per line..."
        )
        
        if st.button("Analyze Batch"):
            if batch_input:
                texts = [t.strip() for t in batch_input.split("\n") if t.strip()]
                
                results_data = []
                progress_bar = st.progress(0)
                
                for i, text in enumerate(texts):
                    result = predict_sentiment(classifier, text)
                    if result:
                        results_data.append({
                            "Text": text[:50] + "..." if len(text) > 50 else text,
                            "Sentiment": result["label"],
                            "Confidence": f"{result['score']:.1%}",
                            "Emoji": result["emoji"]
                        })
                    progress_bar.progress((i + 1) / len(texts))
                
                if results_data:
                    import pandas as pd
                    df = pd.DataFrame(results_data)
                    st.dataframe(df, use_container_width=True)
                    
                    # Summary
                    pos_count = sum(1 for r in results_data if r["Sentiment"] == "positive")
                    neg_count = len(results_data) - pos_count
                    st.markdown(
                        f"**Summary:** {pos_count} positive, {neg_count} negative"
                    )
            else:
                st.warning("Please enter texts to analyze.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: gray; font-size: 12px;">
        Built with ❤️ using Streamlit and Hugging Face Transformers<br>
        <a href="https://github.com/YOUR_USERNAME/crosslingual-sentiment">GitHub</a> | 
        <a href="https://huggingface.co/YOUR_USERNAME/crosslingual-sentiment-model">Model</a>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
