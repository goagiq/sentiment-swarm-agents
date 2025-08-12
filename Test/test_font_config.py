#!/usr/bin/env python3
"""
Test script for font configuration with Chinese characters.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config.font_config import configure_font_for_language, test_font_support
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

def test_chinese_font():
    """Test Chinese font configuration."""
    print("Testing Chinese font configuration...")
    
    # Test Chinese text
    chinese_text = "中文测试文字"
    
    # Test font configuration
    font_set = configure_font_for_language("zh")
    print(f"Font configured for Chinese: {font_set}")
    
    # Get current font
    current_font = plt.rcParams.get('font.sans-serif', ['DejaVu Sans'])[0]
    print(f"Current font: {current_font}")
    
    # Test font support
    test_result = test_font_support(chinese_text, "zh")
    print(f"Font test result: {test_result}")
    
    # Create a simple plot to test rendering
    try:
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, chinese_text, fontsize=24, ha='center', va='center')
        plt.title("Chinese Font Test")
        plt.savefig("Test/chinese_font_test.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("✅ Chinese font test successful - image saved as Test/chinese_font_test.png")
    except Exception as e:
        print(f"❌ Chinese font test failed: {e}")

def test_russian_font():
    """Test Russian font configuration."""
    print("\nTesting Russian font configuration...")
    
    # Test Russian text
    russian_text = "Русский тест"
    
    # Test font configuration
    font_set = configure_font_for_language("ru")
    print(f"Font configured for Russian: {font_set}")
    
    # Get current font
    current_font = plt.rcParams.get('font.sans-serif', ['DejaVu Sans'])[0]
    print(f"Current font: {current_font}")
    
    # Test font support
    test_result = test_font_support(russian_text, "ru")
    print(f"Font test result: {test_result}")
    
    # Create a simple plot to test rendering
    try:
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, russian_text, fontsize=24, ha='center', va='center')
        plt.title("Russian Font Test")
        plt.savefig("Test/russian_font_test.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("✅ Russian font test successful - image saved as Test/russian_font_test.png")
    except Exception as e:
        print(f"❌ Russian font test failed: {e}")

def list_available_fonts():
    """List available fonts on the system."""
    print("\nAvailable fonts:")
    fonts = [f.name for f in fm.fontManager.ttflist]
    chinese_fonts = [f for f in fonts if any(keyword in f.lower() for keyword in ['chinese', 'simhei', 'yahei', 'simsun', 'pingfang', 'hiragino'])]
    russian_fonts = [f for f in fonts if any(keyword in f.lower() for keyword in ['cyrillic', 'russian', 'arial', 'times'])]
    
    print(f"Chinese fonts found: {len(chinese_fonts)}")
    for font in chinese_fonts[:10]:  # Show first 10
        print(f"  - {font}")
    
    print(f"\nRussian fonts found: {len(russian_fonts)}")
    for font in russian_fonts[:10]:  # Show first 10
        print(f"  - {font}")

if __name__ == "__main__":
    print("Font Configuration Test")
    print("=" * 50)
    
    list_available_fonts()
    test_chinese_font()
    test_russian_font()
    
    print("\nTest completed!")
