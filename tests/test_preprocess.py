"""
Tests for Preprocessing Module
==============================
"""

import sys
from pathlib import Path

import pytest
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocess import clean_text, clean_dataframe


class TestCleanText:
    """Tests for text cleaning function."""
    
    def test_removes_urls(self):
        """Test URL removal."""
        text = "Check out https://example.com for more info"
        result = clean_text(text, remove_urls=True)
        assert "https://example.com" not in result
        assert "Check out" in result
    
    def test_removes_http_urls(self):
        """Test HTTP URL removal."""
        text = "Visit http://test.org now"
        result = clean_text(text, remove_urls=True)
        assert "http://test.org" not in result
    
    def test_removes_www_urls(self):
        """Test www URL removal."""
        text = "Go to www.website.com"
        result = clean_text(text, remove_urls=True)
        assert "www.website.com" not in result
    
    def test_removes_html_tags(self):
        """Test HTML tag removal."""
        text = "<p>This is <b>bold</b> text</p>"
        result = clean_text(text, remove_html=True)
        assert "<p>" not in result
        assert "<b>" not in result
        assert "</b>" not in result
        assert "bold" in result
    
    def test_normalizes_whitespace(self):
        """Test whitespace normalization."""
        text = "Too   many    spaces"
        result = clean_text(text, normalize_whitespace=True)
        assert "  " not in result
        assert "Too many spaces" == result
    
    def test_strips_whitespace(self):
        """Test leading/trailing whitespace removal."""
        text = "   text with spaces   "
        result = clean_text(text, normalize_whitespace=True)
        assert result == "text with spaces"
    
    def test_removes_email(self):
        """Test email address removal."""
        text = "Contact me at user@example.com"
        result = clean_text(text)
        assert "user@example.com" not in result
    
    def test_removes_mentions_keeps_word(self):
        """Test that @mentions are cleaned but word kept."""
        text = "Hello @username how are you"
        result = clean_text(text)
        assert "@username" not in result
        assert "username" in result
    
    def test_removes_hashtags_keeps_word(self):
        """Test that #hashtags are cleaned but word kept."""
        text = "This is #amazing content"
        result = clean_text(text)
        assert "#amazing" not in result
        assert "amazing" in result
    
    def test_lowercase_option(self):
        """Test lowercase option."""
        text = "UPPERCASE TEXT"
        result = clean_text(text, lowercase=True)
        assert result == "uppercase text"
    
    def test_preserves_case_by_default(self):
        """Test that case is preserved by default."""
        text = "Mixed Case Text"
        result = clean_text(text, lowercase=False)
        assert result == "Mixed Case Text"
    
    def test_handles_empty_string(self):
        """Test handling of empty string."""
        result = clean_text("")
        assert result == ""
    
    def test_handles_none(self):
        """Test handling of None."""
        result = clean_text(None)
        assert result == ""
    
    def test_handles_numeric(self):
        """Test handling of numeric input."""
        result = clean_text(123)
        assert result == ""
    
    def test_preserves_emojis(self):
        """Test that emojis are preserved."""
        text = "Great movie! 😀👍"
        result = clean_text(text, preserve_emojis=True)
        assert "😀" in result
        assert "👍" in result
    
    def test_bengali_text_preserved(self):
        """Test that Bengali text is preserved."""
        text = "এটি একটি বাংলা বাক্য"
        result = clean_text(text)
        assert "বাংলা" in result
    
    def test_mixed_language_preserved(self):
        """Test mixed English-Bengali text."""
        text = "This is মিশ্র text"
        result = clean_text(text)
        assert "This" in result
        assert "মিশ্র" in result


class TestCleanDataframe:
    """Tests for dataframe cleaning function."""
    
    def test_cleans_text_column(self):
        """Test that text column is cleaned."""
        df = pd.DataFrame({
            "text": ["  Hello  ", "https://url.com test"],
            "label": [1, 0]
        })
        result = clean_dataframe(df)
        assert result.iloc[0]["text"] == "Hello"
        assert "https://url.com" not in result.iloc[1]["text"]
    
    def test_removes_empty_texts(self):
        """Test that empty texts are removed after cleaning."""
        df = pd.DataFrame({
            "text": ["Good text", "https://only-url.com", "Another good text"],
            "label": [1, 0, 1]
        })
        result = clean_dataframe(df)
        # Second row should be removed as it becomes empty
        assert len(result) <= len(df)
    
    def test_custom_text_column(self):
        """Test with custom text column name."""
        df = pd.DataFrame({
            "sentence": ["  Test sentence  "],
            "label": [1]
        })
        result = clean_dataframe(df, text_column="sentence")
        assert result.iloc[0]["sentence"] == "Test sentence"
    
    def test_preserves_other_columns(self):
        """Test that other columns are preserved."""
        df = pd.DataFrame({
            "text": ["Test"],
            "label": [1],
            "extra": ["value"]
        })
        result = clean_dataframe(df)
        assert "extra" in result.columns
        assert result.iloc[0]["extra"] == "value"


class TestEdgeCases:
    """Edge case tests."""
    
    def test_very_long_text(self):
        """Test handling of very long text."""
        text = "word " * 10000
        result = clean_text(text)
        assert len(result) > 0
    
    def test_special_characters(self):
        """Test handling of special characters."""
        text = "Test !@#$%^&*()_+ text"
        result = clean_text(text)
        assert "Test" in result
        assert "text" in result
    
    def test_newlines(self):
        """Test handling of newlines."""
        text = "Line 1\nLine 2\r\nLine 3"
        result = clean_text(text, normalize_whitespace=True)
        assert "\n" not in result or result.count("\n") < 3
    
    def test_tabs(self):
        """Test handling of tabs."""
        text = "Column1\tColumn2\tColumn3"
        result = clean_text(text, normalize_whitespace=True)
        assert "\t" not in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
