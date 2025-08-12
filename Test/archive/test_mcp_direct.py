#!/usr/bin/env python3
"""
Test MCP PDF processing directly using the MCP client.
"""

import os
import sys
import asyncio
import json
from datetime import datetime

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.config.language_config import LanguageConfigFactory


async def test_mcp_pdf_processing():
    """Test PDF processing using MCP client directly."""
    
    print("🧪 Testing MCP PDF Processing Directly")
    print("=" * 50)
    
    try:
        # Import the MCP server
        from src.core.mcp_server import OptimizedMCPServer
        
        print("🔧 Initializing MCP server...")
        mcp_server = OptimizedMCPServer()
        
        if not mcp_server.mcp:
            print("❌ MCP server not available")
            return False
        
        print("✅ MCP server initialized successfully")
        
        # Test the process_pdf_enhanced_multilingual tool
        print("\n🔬 Testing process_pdf_enhanced_multilingual tool...")
        
        pdf_path = "data/Classical Chinese Sample 22208_0_8.pdf"
        
        if not os.path.exists(pdf_path):
            print(f"❌ PDF file not found: {pdf_path}")
            return False
        
        print(f"📄 Processing PDF: {pdf_path}")
        
        # Call the MCP tool directly
        result = await mcp_server.mcp.process_pdf_enhanced_multilingual(
            pdf_path=pdf_path,
            language="zh",
            generate_report=True,
            output_path=None
        )
        
        print("✅ PDF processing completed successfully!")
        
        # Display results
        print("\n" + "=" * 50)
        print("📋 PROCESSING RESULTS")
        print("=" * 50)
        
        if isinstance(result, dict):
            for key, value in result.items():
                print(f"   - {key}: {value}")
        else:
            print(f"   - Result: {result}")
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f"Results/mcp_direct_test_results_{timestamp}.json"
        
        os.makedirs("Results", exist_ok=True)
        
        with open(results_file, 'w', encoding='utf-8') as f:
            if isinstance(result, dict):
                json.dump(result, f, indent=2, ensure_ascii=False)
            else:
                json.dump({"result": str(result)}, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 Results saved to: {results_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing MCP PDF processing: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_chinese_config():
    """Test Chinese language configuration."""
    print("\n🏛️ Testing Chinese Language Configuration")
    print("=" * 50)
    
    try:
        chinese_config = LanguageConfigFactory.get_config("zh")
        print("✅ Chinese language configuration loaded")
        
        # Check Classical Chinese patterns
        if hasattr(chinese_config, 'classical_patterns'):
            print(f"✅ Classical Chinese patterns: {len(chinese_config.classical_patterns)} categories")
        
        if hasattr(chinese_config, 'is_classical_chinese'):
            print("✅ Classical Chinese detection method available")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing Chinese config: {e}")
        return False


async def main():
    """Main test function."""
    print("🧪 MCP Direct PDF Processing Test")
    print("=" * 60)
    
    # Test Chinese configuration
    config_success = await test_chinese_config()
    
    if config_success:
        # Test MCP PDF processing
        mcp_success = await test_mcp_pdf_processing()
        
        if mcp_success:
            print("\n🎉 All tests completed successfully!")
        else:
            print("\n❌ MCP PDF processing test failed.")
    else:
        print("\n❌ Chinese configuration test failed.")


if __name__ == "__main__":
    asyncio.run(main())
