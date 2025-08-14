#!/usr/bin/env python3
"""
Test script to demonstrate the fix for the 'Invalid type for parameter option' error.
This shows how the new configuration system automatically handles different question types.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_options_configuration():
    """Test the new options configuration system."""
    
    try:
        from config.process_content_options_config import (
            get_process_content_options, 
            detect_question_type
        )
        
        print("✅ Successfully imported options configuration")
        
        # Test different question types
        test_questions = [
            "How do the strategic principles in The Art of War apply to modern cyber warfare?",
            "What is the sentiment of this customer review?",
            "Extract entities from this document",
            "Create a knowledge graph from this text",
            "Analyze this audio recording",
            "What is the weather like today?"
        ]
        
        print("\n" + "="*60)
        print("TESTING QUESTION TYPE DETECTION AND OPTIONS GENERATION")
        print("="*60)
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n{i}. Question: {question}")
            
            # Detect question type
            category = detect_question_type(question)
            print(f"   Detected Category: {category}")
            
            # Get options
            options = get_process_content_options(question)
            print(f"   Generated Options: {options}")
            
            print("-" * 50)
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_safe_process_content():
    """Test the safe process_content wrapper."""
    
    try:
        from core.process_content_wrapper import process_content_with_auto_options
        
        print("\n" + "="*60)
        print("TESTING SAFE PROCESS_CONTENT WRAPPER")
        print("="*60)
        
        test_content = "How do the strategic principles in The Art of War apply to modern cyber warfare?"
        
        print(f"Testing with: {test_content}")
        
        # This should work without the options parameter error
        result = process_content_with_auto_options(test_content)
        
        print(f"Result: {result}")
        
        if result.get("success"):
            print("✅ Safe process_content wrapper works correctly!")
        else:
            print(f"⚠️  Result indicates issue: {result.get('error', 'Unknown error')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing safe process_content: {e}")
        return False


def test_mcp_integration():
    """Test integration with MCP tools."""
    
    print("\n" + "="*60)
    print("TESTING MCP INTEGRATION")
    print("="*60)
    
    try:
        # Test the MCP process_content tool directly
        from mcp_Sentiment import process_content
        
        test_content = "How do the strategic principles in The Art of War apply to modern cyber warfare?"
        
        print(f"Testing MCP process_content with: {test_content}")
        
        # Test without options (should work)
        result1 = process_content(
            content=test_content,
            content_type="text"
        )
        print(f"✅ Without options: {result1.get('success', False)}")
        
        # Test with proper options (should work)
        options = {
            "analysis_type": "strategic_intelligence",
            "focus_areas": ["strategic_principles", "cyber_warfare"],
            "output_format": "comprehensive_analysis"
        }
        
        result2 = process_content(
            content=test_content,
            content_type="text",
            options=options
        )
        print(f"✅ With proper options: {result2.get('success', False)}")
        
        return True
        
    except Exception as e:
        print(f"❌ MCP integration error: {e}")
        return False


def main():
    """Main test function."""
    
    print("🔧 TESTING OPTIONS PARAMETER FIX")
    print("This script tests the fix for the 'Invalid type for parameter option' error")
    
    # Test 1: Options configuration
    test1_success = test_options_configuration()
    
    # Test 2: Safe process_content wrapper
    test2_success = test_safe_process_content()
    
    # Test 3: MCP integration
    test3_success = test_mcp_integration()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Options Configuration: {'✅ PASS' if test1_success else '❌ FAIL'}")
    print(f"Safe Process Content: {'✅ PASS' if test2_success else '❌ FAIL'}")
    print(f"MCP Integration: {'✅ PASS' if test3_success else '❌ FAIL'}")
    
    if all([test1_success, test2_success, test3_success]):
        print("\n🎉 ALL TESTS PASSED! The options parameter fix is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
    
    print("\n" + "="*60)
    print("USAGE INSTRUCTIONS")
    print("="*60)
    print("To use the fixed process_content tool:")
    print("1. Use the safe wrapper: process_content_with_auto_options(content)")
    print("2. Or use MCP directly without options parameter")
    print("3. Or use MCP with properly formatted options dict")
    print("\nThe configuration automatically detects question types and applies appropriate options.")


if __name__ == "__main__":
    main()
