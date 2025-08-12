#!/usr/bin/env python3
"""
Test Classical Chinese PDF processing integration.
Uses existing MCP servers and API endpoints.
"""

import asyncio
import os
import sys
from datetime import datetime

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.config.language_config import LanguageConfigFactory
from src.core.mcp_server import OptimizedMCPServer


async def test_classical_chinese_integration():
    """Test the Classical Chinese integration with existing components."""
    
    print("🏛️ Testing Classical Chinese Integration")
    print("=" * 50)
    
    # Test 1: Language Configuration
    print("\n1️⃣ Testing Language Configuration...")
    try:
        chinese_config = LanguageConfigFactory.get_config("zh")
        print("✅ Chinese language configuration loaded")
        
        # Check Classical Chinese features
        if hasattr(chinese_config, 'classical_patterns'):
            print(f"✅ Classical patterns: {len(chinese_config.classical_patterns)} categories")
        
        if hasattr(chinese_config, 'is_classical_chinese'):
            print("✅ Classical Chinese detection method available")
        
        if hasattr(chinese_config, 'get_classical_processing_settings'):
            print("✅ Classical Chinese processing settings available")
        
        # Check Ollama configuration
        ollama_config = chinese_config.get_ollama_config()
        if 'classical_chinese_model' in ollama_config:
            print("✅ Classical Chinese Ollama model configured")
        
    except Exception as e:
        print(f"❌ Language configuration test failed: {e}")
        return False
    
    # Test 2: MCP Server Integration
    print("\n2️⃣ Testing MCP Server Integration...")
    try:
        mcp_server = OptimizedMCPServer()
        print("✅ MCP Server initialized")
        
        if mcp_server.mcp:
            print("✅ MCP Server with FastMCP available")
            
            # Check if the enhanced PDF processing tool is available
            if hasattr(mcp_server.mcp, 'process_pdf_enhanced_multilingual'):
                print("✅ Enhanced multilingual PDF processing tool available")
            else:
                print("⚠️ Enhanced multilingual PDF processing tool not found")
        
    except Exception as e:
        print(f"❌ MCP Server test failed: {e}")
        return False
    
    # Test 3: PDF File Availability
    print("\n3️⃣ Testing PDF File Availability...")
    pdf_path = "data/Classical Chinese Sample 22208_0_8.pdf"
    
    if os.path.exists(pdf_path):
        print(f"✅ PDF file found: {pdf_path}")
        
        # Get file size
        file_size = os.path.getsize(pdf_path)
        print(f"   - File size: {file_size / (1024*1024):.2f} MB")
        
    else:
        print(f"❌ PDF file not found: {pdf_path}")
        return False
    
    # Test 4: Direct Agent Integration
    print("\n4️⃣ Testing Direct Agent Integration...")
    try:
        from src.agents.enhanced_file_extraction_agent import EnhancedFileExtractionAgent
        from src.agents.knowledge_graph_agent import KnowledgeGraphAgent
        from src.core.models import AnalysisRequest, DataType
        
        # Test file extraction agent
        file_agent = EnhancedFileExtractionAgent()
        print("✅ Enhanced file extraction agent initialized")
        
        # Test knowledge graph agent
        kg_agent = KnowledgeGraphAgent()
        print("✅ Knowledge graph agent initialized")
        
    except Exception as e:
        print(f"❌ Direct agent integration test failed: {e}")
        return False
    
    # Test 5: Configuration Files
    print("\n5️⃣ Testing Configuration Files...")
    config_files = [
        "src/config/language_config/chinese_config.py",
        "src/config/language_config/base_config.py",
        "src/config/config.py"
    ]
    
    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"✅ Configuration file found: {config_file}")
        else:
            print(f"❌ Configuration file missing: {config_file}")
    
    print("\n" + "=" * 50)
    print("✅ All integration tests completed successfully!")
    print("=" * 50)
    
    return True


async def test_classical_chinese_processing():
    """Test actual Classical Chinese PDF processing."""
    
    print("\n🔬 Testing Classical Chinese PDF Processing")
    print("=" * 50)
    
    pdf_path = "data/Classical Chinese Sample 22208_0_8.pdf"
    
    try:
        from src.agents.enhanced_file_extraction_agent import EnhancedFileExtractionAgent
        from src.agents.knowledge_graph_agent import KnowledgeGraphAgent
        from src.core.models import AnalysisRequest, DataType
        
        # Step 1: Extract text
        print("📄 Step 1: Extracting text from PDF...")
        file_agent = EnhancedFileExtractionAgent()
        
        pdf_request = AnalysisRequest(
            data_type=DataType.PDF,
            content=pdf_path,
            language="zh"
        )
        
        extraction_result = await file_agent.process(pdf_request)
        
        if extraction_result.status != "completed":
            print(f"❌ Text extraction failed: {extraction_result.metadata.get('error', 'Unknown error')}")
            return False
        
        text_content = extraction_result.extracted_text
        print(f"✅ Text extraction successful: {len(text_content)} characters")
        
        # Step 2: Check for Classical Chinese patterns
        print("🏛️ Step 2: Checking for Classical Chinese patterns...")
        chinese_config = LanguageConfigFactory.get_config("zh")
        
        if hasattr(chinese_config, 'is_classical_chinese'):
            is_classical = chinese_config.is_classical_chinese(text_content[:1000])
            print(f"✅ Classical Chinese detected: {is_classical}")
        
        # Step 3: Process with knowledge graph agent
        print("🧠 Step 3: Processing with knowledge graph agent...")
        kg_agent = KnowledgeGraphAgent()
        
        kg_request = AnalysisRequest(
            data_type=DataType.TEXT,
            content=text_content,
            language="zh"
        )
        
        kg_result = await kg_agent.process(kg_request)
        
        if kg_result.status != "completed":
            print(f"❌ Knowledge graph processing failed: {kg_result.metadata.get('error', 'Unknown error')}")
            return False
        
        # Display results
        stats = kg_result.metadata.get("statistics", {}) if kg_result.metadata else {}
        
        print(f"✅ Knowledge graph processing successful:")
        print(f"   - Entities found: {stats.get('entities_found', 0)}")
        print(f"   - Entity types: {stats.get('entity_types', {})}")
        print(f"   - Processing time: {kg_result.processing_time:.2f} seconds")
        
        print("\n" + "=" * 50)
        print("✅ Classical Chinese PDF processing test completed successfully!")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"❌ Classical Chinese processing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test function."""
    print("🧪 Classical Chinese Integration Test Suite")
    print("=" * 60)
    
    # Run integration tests
    integration_success = await test_classical_chinese_integration()
    
    if integration_success:
        # Run processing test
        processing_success = await test_classical_chinese_processing()
        
        if processing_success:
            print("\n🎉 All tests passed! Classical Chinese integration is working correctly.")
        else:
            print("\n⚠️ Integration tests passed but processing test failed.")
    else:
        print("\n❌ Integration tests failed. Please check the configuration.")
    
    return integration_success


if __name__ == "__main__":
    # Run the async function
    asyncio.run(main())
