#!/usr/bin/env python3
"""
Debug script to test Russian language detection.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config.language_specific_config import detect_primary_language, detect_language_from_text

def test_language_detection():
    """Test language detection with Russian text."""
    
    # Sample Russian text from the PDF
    russian_text = """
    Первый круг
    Russian Full Circle
    A First-Year Russian Textbook
    Donna Oliver
    Beloit College
    with Edie Furniss
    The Pennsylvania State University
    New Haven and London
    From 'Russian Full Circle'
    """
    
    print("Testing language detection...")
    print(f"Text sample: {russian_text[:100]}...")
    
    # Test primary language detection
    primary_lang = detect_primary_language(russian_text)
    print(f"Primary language detected: {primary_lang}")
    
    # Test pattern-based detection
    pattern_lang = detect_language_from_text(russian_text)
    print(f"Pattern-based language detected: {pattern_lang}")
    
    # Count characters
    russian_chars = len([c for c in russian_text if '\u0400' <= c <= '\u04FF'])
    chinese_chars = len([c for c in russian_text if '\u4e00' <= c <= '\u9fff'])
    english_words = len([w for w in russian_text.split() if w.isalpha() and w.isascii()])
    
    print(f"Russian characters: {russian_chars}")
    print(f"Chinese characters: {chinese_chars}")
    print(f"English words: {english_words}")
    print(f"Total characters: {len(russian_text)}")
    
    # Calculate ratios
    total_chars = len(russian_text.replace(' ', ''))
    if total_chars > 0:
        russian_ratio = russian_chars / total_chars
        chinese_ratio = chinese_chars / total_chars
        print(f"Russian ratio: {russian_ratio:.3f}")
        print(f"Chinese ratio: {chinese_ratio:.3f}")
        
        if russian_ratio > 0.1:
            print("✓ Should detect as Russian")
        elif chinese_ratio > 0.1:
            print("✓ Should detect as Chinese")
        else:
            print("✓ Should detect as English")

if __name__ == "__main__":
    test_language_detection()
