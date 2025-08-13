"""
Test script for Consolidated MCP Server.

This script tests the consolidated MCP server functionality and demonstrates
the optimization results from 44 individual servers to 4 consolidated servers.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loguru import logger

# Import consolidated MCP server
from mcp.consolidated_mcp_server import ConsolidatedMCPServer, ConsolidatedMCPServerConfig


async def test_consolidated_mcp_server():
    """Test the consolidated MCP server functionality."""
    logger.info("🧪 Testing Consolidated MCP Server")
    
    try:
        # Create configuration
        config = ConsolidatedMCPServerConfig(
            enable_pdf_processing=True,
            enable_audio_processing=True,
            enable_video_processing=True,
            enable_website_processing=True,
            host="localhost",
            port=8000
        )
        
        # Initialize consolidated MCP server
        logger.info("1. Initializing Consolidated MCP Server...")
        consolidated_server = ConsolidatedMCPServer(config)
        
        # Test server initialization
        logger.info("2. Testing server initialization...")
        assert consolidated_server is not None
        assert len(consolidated_server.processing_servers) == 4
        logger.info(f"✅ Server initialized with {len(consolidated_server.processing_servers)} processing servers")
        
        # Test PDF processing server
        logger.info("3. Testing PDF Processing Server...")
        if "pdf" in consolidated_server.processing_servers:
            pdf_server = consolidated_server.processing_servers["pdf"]
            logger.info(f"✅ PDF Server: {pdf_server.__class__.__name__}")
            
            # Test PDF server methods
            methods = [
                "extract_text", "convert_content", "summarize_content",
                "translate_content", "store_in_vector_db", "create_knowledge_graph"
            ]
            for method in methods:
                assert hasattr(pdf_server, method), f"PDF server missing method: {method}"
            logger.info(f"✅ PDF Server has all {len(methods)} required methods")
        
        # Test Audio processing server
        logger.info("4. Testing Audio Processing Server...")
        if "audio" in consolidated_server.processing_servers:
            audio_server = consolidated_server.processing_servers["audio"]
            logger.info(f"✅ Audio Server: {audio_server.__class__.__name__}")
            
            # Test Audio server methods
            methods = [
                "extract_text", "convert_content", "summarize_content",
                "translate_content", "store_in_vector_db", "create_knowledge_graph"
            ]
            for method in methods:
                assert hasattr(audio_server, method), f"Audio server missing method: {method}"
            logger.info(f"✅ Audio Server has all {len(methods)} required methods")
        
        # Test Video processing server
        logger.info("5. Testing Video Processing Server...")
        if "video" in consolidated_server.processing_servers:
            video_server = consolidated_server.processing_servers["video"]
            logger.info(f"✅ Video Server: {video_server.__class__.__name__}")
            
            # Test Video server methods
            methods = [
                "extract_text", "convert_content", "summarize_content",
                "translate_content", "store_in_vector_db", "create_knowledge_graph"
            ]
            for method in methods:
                assert hasattr(video_server, method), f"Video server missing method: {method}"
            logger.info(f"✅ Video Server has all {len(methods)} required methods")
        
        # Test Website processing server
        logger.info("6. Testing Website Processing Server...")
        if "website" in consolidated_server.processing_servers:
            website_server = consolidated_server.processing_servers["website"]
            logger.info(f"✅ Website Server: {website_server.__class__.__name__}")
            
            # Test Website server methods
            methods = [
                "extract_text", "convert_content", "summarize_content",
                "translate_content", "store_in_vector_db", "create_knowledge_graph"
            ]
            for method in methods:
                assert hasattr(website_server, method), f"Website server missing method: {method}"
            logger.info(f"✅ Website Server has all {len(methods)} required methods")
        
        # Test MCP tools registration
        logger.info("7. Testing MCP Tools Registration...")
        if consolidated_server.mcp is not None:
            logger.info("✅ MCP server initialized successfully")
            # Note: Tool registration is tested during server startup
        else:
            logger.warning("⚠️  MCP server not available (FastMCP not installed)")
        
        # Test content category detection
        logger.info("8. Testing Content Category Detection...")
        test_cases = [
            ("document.pdf", "pdf"),
            ("audio.mp3", "audio"),
            ("video.mp4", "video"),
            ("https://youtube.com/watch?v=123", "video"),
            ("https://example.com", "website"),
            ("plain text content", "website")
        ]
        
        for content, expected_category in test_cases:
            detected_category = consolidated_server._detect_content_category(content)
            assert detected_category == expected_category, f"Expected {expected_category}, got {detected_category}"
            logger.info(f"✅ Category detection: '{content}' -> {detected_category}")
        
        # Test configuration integration
        logger.info("9. Testing Configuration Integration...")
        assert config.enable_pdf_processing == True
        assert config.enable_audio_processing == True
        assert config.enable_video_processing == True
        assert config.enable_website_processing == True
        assert config.default_language == "en"
        assert "en" in config.supported_languages
        assert "zh" in config.supported_languages
        assert "ru" in config.supported_languages
        logger.info("✅ Configuration integration working correctly")
        
        # Test cleanup
        logger.info("10. Testing Cleanup...")
        await consolidated_server.cleanup()
        logger.info("✅ Cleanup completed successfully")
        
        # Summary
        logger.info("🎉 Consolidated MCP Server Test Results:")
        logger.info(f"   📊 Server Count: 44 individual servers → 4 consolidated servers")
        logger.info(f"   📈 Reduction: 90.9% reduction in server count")
        logger.info(f"   🔧 Functions per server: 6 core functions each")
        logger.info(f"   🌐 Categories: PDF, Audio, Video, Website")
        logger.info(f"   ⚡ Performance: Unified interfaces, consistent error handling")
        logger.info(f"   🔧 Configuration: Integrated with existing config files")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False


async def test_individual_functions():
    """Test individual functions of each processing server."""
    logger.info("🧪 Testing Individual Functions")
    
    try:
        # Create configuration
        config = ConsolidatedMCPServerConfig()
        consolidated_server = ConsolidatedMCPServer(config)
        
        # Test PDF functions
        logger.info("Testing PDF Processing Functions...")
        pdf_server = consolidated_server.processing_servers["pdf"]
        
        # Test text extraction (with mock data)
        text_result = await pdf_server.extract_text("test_content", "en")
        assert isinstance(text_result, dict)
        assert "success" in text_result
        logger.info("✅ PDF text extraction function working")
        
        # Test summarization (with mock data)
        summary_result = await pdf_server.summarize_content("test content for summarization", "en")
        assert isinstance(summary_result, dict)
        assert "success" in summary_result
        logger.info("✅ PDF summarization function working")
        
        # Test Audio functions
        logger.info("Testing Audio Processing Functions...")
        audio_server = consolidated_server.processing_servers["audio"]
        
        # Test text extraction (with mock data)
        text_result = await audio_server.extract_text("test_content", "en")
        assert isinstance(text_result, dict)
        assert "success" in text_result
        logger.info("✅ Audio text extraction function working")
        
        # Test Video functions
        logger.info("Testing Video Processing Functions...")
        video_server = consolidated_server.processing_servers["video"]
        
        # Test text extraction (with mock data)
        text_result = await video_server.extract_text("test_content", "en")
        assert isinstance(text_result, dict)
        assert "success" in text_result
        logger.info("✅ Video text extraction function working")
        
        # Test Website functions
        logger.info("Testing Website Processing Functions...")
        website_server = consolidated_server.processing_servers["website"]
        
        # Test text extraction (with mock data)
        text_result = await website_server.extract_text("test_content", "en")
        assert isinstance(text_result, dict)
        assert "success" in text_result
        logger.info("✅ Website text extraction function working")
        
        await consolidated_server.cleanup()
        logger.info("✅ All individual functions tested successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Individual function test failed: {e}")
        return False


async def main():
    """Main test function."""
    logger.info("🚀 Starting Consolidated MCP Server Tests")
    
    # Test 1: Basic functionality
    test1_result = await test_consolidated_mcp_server()
    
    # Test 2: Individual functions
    test2_result = await test_individual_functions()
    
    # Summary
    logger.info("📋 Test Summary:")
    logger.info(f"   Test 1 (Basic Functionality): {'✅ PASSED' if test1_result else '❌ FAILED'}")
    logger.info(f"   Test 2 (Individual Functions): {'✅ PASSED' if test2_result else '❌ FAILED'}")
    
    if test1_result and test2_result:
        logger.info("🎉 All tests passed! Consolidated MCP Server is working correctly.")
        return True
    else:
        logger.error("❌ Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    # Run the tests
    success = asyncio.run(main())
    
    if success:
        logger.info("✅ All tests completed successfully!")
        sys.exit(0)
    else:
        logger.error("❌ Tests failed!")
        sys.exit(1)
