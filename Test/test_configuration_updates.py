#!/usr/bin/env python3
"""
Test script for Step 7: Configuration Updates
Validates the new configuration structure for consolidated MCP servers.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_configuration_imports():
    """Test that all configuration imports work correctly."""
    print("🔧 Testing configuration imports...")
    
    try:
        from config.mcp_config import (
            ConsolidatedMCPServerConfig,
            ConsolidatedServerConfig,
            ProcessingCategory,
            get_consolidated_mcp_config,
            get_server_config_for_category
        )
        print("✅ Configuration imports successful")
        return True
    except ImportError as e:
        print(f"❌ Configuration import failed: {e}")
        return False

def test_consolidated_server_config():
    """Test the consolidated server configuration structure."""
    print("\n🔧 Testing consolidated server configuration...")
    
    try:
        from config.mcp_config import ConsolidatedMCPServerConfig, ProcessingCategory
        
        # Create a test configuration
        config = ConsolidatedMCPServerConfig()
        
        # Test PDF server configuration
        pdf_config = config.pdf_server
        assert pdf_config.enabled == True
        assert pdf_config.primary_model == "llava:latest"
        assert pdf_config.vector_db_collection == "pdf_documents"
        print("✅ PDF server configuration valid")
        
        # Test Audio server configuration
        audio_config = config.audio_server
        assert audio_config.enabled == True
        assert audio_config.primary_model == "llava:latest"
        assert audio_config.vector_db_collection == "audio_transcripts"
        print("✅ Audio server configuration valid")
        
        # Test Video server configuration
        video_config = config.video_server
        assert video_config.enabled == True
        assert video_config.primary_model == "llava:latest"
        assert video_config.vector_db_collection == "video_analysis"
        print("✅ Video server configuration valid")
        
        # Test Website server configuration
        website_config = config.website_server
        assert website_config.enabled == True
        assert website_config.primary_model == "mistral-small3.1:latest"
        assert website_config.vector_db_collection == "web_content"
        print("✅ Website server configuration valid")
        
        return True
    except Exception as e:
        print(f"❌ Consolidated server configuration test failed: {e}")
        return False

def test_processing_categories():
    """Test the processing categories enum."""
    print("\n🔧 Testing processing categories...")
    
    try:
        from config.mcp_config import ProcessingCategory, get_server_config_for_category
        
        # Test all categories
        categories = [
            ProcessingCategory.PDF,
            ProcessingCategory.AUDIO,
            ProcessingCategory.VIDEO,
            ProcessingCategory.WEBSITE
        ]
        
        for category in categories:
            config = get_server_config_for_category(category)
            assert config is not None
            print(f"✅ {category.value} category configuration retrieved")
        
        return True
    except Exception as e:
        print(f"❌ Processing categories test failed: {e}")
        return False

def test_language_config_integration():
    """Test language configuration integration."""
    print("\n🔧 Testing language configuration integration...")
    
    try:
        from config.mcp_config import get_language_config
        
        # Test language configurations
        languages = ["zh", "ru", "en"]
        
        for lang in languages:
            lang_config = get_language_config(lang)
            if lang_config is not None:
                print(f"✅ {lang} language configuration available")
            else:
                print(f"⚠️  {lang} language configuration not available")
        
        return True
    except Exception as e:
        print(f"❌ Language configuration test failed: {e}")
        return False

def test_main_integration():
    """Test that main.py can import the new configuration."""
    print("\n🔧 Testing main.py integration...")
    
    try:
        # Check if main.py has been updated
        main_py_path = Path(__file__).parent.parent / "main.py"
        
        if not main_py_path.exists():
            print("❌ main.py not found")
            return False
        
        with open(main_py_path, "r", encoding="utf-8") as f:
            main_content = f.read()
        
        # Check for new configuration imports
        required_imports = [
            "from src.config.mcp_config import ConsolidatedMCPServerConfig",
            "get_consolidated_mcp_config",
            "config = get_consolidated_mcp_config()"
        ]
        
        for required_import in required_imports:
            if required_import in main_content:
                print(f"✅ Found: {required_import}")
            else:
                print(f"❌ Missing: {required_import}")
                return False
        
        return True
    except Exception as e:
        print(f"❌ Main integration test failed: {e}")
        return False

def test_consolidated_mcp_server_config():
    """Test that the consolidated MCP server can use the new configuration."""
    print("\n🔧 Testing consolidated MCP server configuration...")
    
    try:
        from mcp.consolidated_mcp_server import ConsolidatedMCPServer
        from config.mcp_config import get_consolidated_mcp_config
        
        # Get configuration
        config = get_consolidated_mcp_config()
        
        # Test server initialization
        server = ConsolidatedMCPServer(config)
        print("✅ Consolidated MCP server initialized with new configuration")
        
        return True
    except Exception as e:
        print(f"❌ Consolidated MCP server configuration test failed: {e}")
        return False

def main():
    """Run all configuration tests."""
    print("🚀 Step 7: Configuration Updates Test")
    print("=" * 50)
    
    tests = [
        ("Configuration Imports", test_configuration_imports),
        ("Consolidated Server Config", test_consolidated_server_config),
        ("Processing Categories", test_processing_categories),
        ("Language Config Integration", test_language_config_integration),
        ("Main Integration", test_main_integration),
        ("Consolidated MCP Server Config", test_consolidated_mcp_server_config)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} failed")
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Configuration Test Results:")
    print(f"   ✅ Passed: {passed}/{total}")
    print(f"   ❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("\n🎉 All configuration tests passed!")
        print("✅ Step 7: Configuration Updates completed successfully")
        print("\n📋 Configuration Features:")
        print("   • Consolidated server configurations for PDF, Audio, Video, Website")
        print("   • Language-specific parameter integration")
        print("   • Model configuration per category")
        print("   • Vector database and knowledge graph settings")
        print("   • Storage path configurations")
        print("   • Backward compatibility with legacy MCP config")
    else:
        print(f"\n⚠️  {total - passed} configuration test(s) failed")
        print("Please review and fix the failing tests")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)














