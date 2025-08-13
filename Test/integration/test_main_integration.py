#!/usr/bin/env python3
"""
Test script to verify main.py integration without starting the full server.
Tests the Ollama integration and language-specific configurations.
"""

import sys
import os
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import main.py functions
from main import check_ollama_integration, get_optimization_status


def test_main_integration():
    """Test the main.py integration functions."""
    
    print("🧪 Testing Main.py Integration...")
    print("=" * 60)
    print(f"📅 Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Test 1: Check Ollama integration
        print("\n📋 Test 1: Checking Ollama Integration...")
        check_ollama_integration()
        
        # Test 2: Get optimization status
        print("\n📋 Test 2: Getting Optimization Status...")
        opt_status = get_optimization_status()
        
        print("\n📊 Optimization Status Summary:")
        print(f" - Phase 1 Complete: {opt_status.get('phase1_complete', False)}")
        print(f" - Phase 2 Complete: {opt_status.get('phase2_complete', False)}")
        print(f" - Phase 3 Complete: {opt_status.get('phase3_complete', False)}")
        print(f" - Phase 4 Complete: {opt_status.get('phase4_complete', False)}")
        print(f" - Ollama Integration Complete: {opt_status.get('ollama_integration_complete', False)}")
        
        if opt_status.get('languages_supported'):
            print(f" - Languages Supported: {', '.join(opt_status['languages_supported'])}")
        
        if opt_status.get('ollama_services'):
            print(f" - Ollama Services: {', '.join(opt_status['ollama_services'])}")
        
        # Test 3: Check specific Ollama services
        print("\n📋 Test 3: Checking Specific Ollama Services...")
        
        ollama_services = opt_status.get('ollama_services', [])
        required_services = [
            "Strands Framework",
            "generate_text Method", 
            "Language-Specific Configurations"
        ]
        
        for service in required_services:
            if service in ollama_services:
                print(f"✅ {service}: Available")
            else:
                print(f"❌ {service}: Not available")
        
        # Test 4: Check language configurations
        print("\n📋 Test 4: Checking Language Configurations...")
        
        languages_supported = opt_status.get('languages_supported', [])
        required_languages = ['zh', 'en', 'ru']
        
        for lang in required_languages:
            if lang in languages_supported:
                print(f"✅ {lang.upper()}: Supported")
            else:
                print(f"❌ {lang.upper()}: Not supported")
        
        print("\n" + "=" * 60)
        print("✅ Main.py Integration Test Completed!")
        
        # Overall success criteria
        success_criteria = [
            opt_status.get('ollama_integration_complete', False),
            len(ollama_services) >= 3,
            len(languages_supported) >= 3
        ]
        
        if all(success_criteria):
            print("🎉 All integration tests PASSED!")
            print("📋 Main.py is ready for production use")
            return True
        else:
            print("⚠️ Some integration tests need attention")
            print("🔧 Check the specific failures above")
            return False
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_main_integration()
    print(f"\n📅 Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📋 Final Result: {'✅ PASSED' if success else '❌ FAILED'}")
    sys.exit(0 if success else 1)
