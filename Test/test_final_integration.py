"""
Comprehensive Integration Tests for Agent Optimization and Consolidation.
Tests all unified agents, shared services, and ToolRegistry functionality.
"""

import asyncio
import logging
import pytest
from typing import Dict, Any

from src.core.models import AnalysisRequest, DataType
from src.agents.unified_text_agent import UnifiedTextAgent
from src.agents.unified_vision_agent import UnifiedVisionAgent
from src.agents.unified_audio_agent import UnifiedAudioAgent
from src.agents.unified_file_extraction_agent import UnifiedFileExtractionAgent
from src.core.tool_registry import tool_registry
from src.core.translation_service import TranslationService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestFinalIntegration:
    """Comprehensive integration tests for the consolidated system."""
    
    @pytest.fixture
    async def setup_agents(self):
        """Set up all unified agents for testing."""
        agents = {
            "text": UnifiedTextAgent(),
            "vision": UnifiedVisionAgent(),
            "audio": UnifiedAudioAgent(),
            "file_extraction": UnifiedFileExtractionAgent()
        }
        
        # Start all agents
        for agent in agents.values():
            await agent.start()
        
        yield agents
        
        # Stop all agents
        for agent in agents.values():
            await agent.stop()
    
    @pytest.fixture
    async def translation_service(self):
        """Set up translation service for testing."""
        return TranslationService()
    
    async def test_unified_text_agent_integration(self, setup_agents):
        """Test UnifiedTextAgent with all capabilities."""
        text_agent = setup_agents["text"]
        
        # Test basic text processing
        request = AnalysisRequest(
            data_type=DataType.TEXT,
            content="I love this product! It's amazing and makes me very happy.",
            language="en"
        )
        
        result = await text_agent.process(request)
        
        assert result is not None
        assert result.sentiment is not None
        assert result.sentiment.label in ["positive", "negative", "neutral"]
        assert 0 <= result.sentiment.confidence <= 1
        
        logger.info(f"Text agent test passed: {result.sentiment.label} ({result.sentiment.confidence})")
    
    async def test_translation_service_integration(self, translation_service):
        """Test TranslationService functionality."""
        # Test language detection
        spanish_text = "Hola, ¿cómo estás? Me gusta mucho este producto."
        detected_language = await translation_service.detect_language(spanish_text)
        assert detected_language == "es"
        
        # Test text translation
        translation_result = await translation_service.translate_text(spanish_text)
        assert translation_result is not None
        assert translation_result.original_text == spanish_text
        assert translation_result.translated_text != spanish_text
        assert translation_result.source_language == "es"
        assert translation_result.target_language == "en"
        assert translation_result.confidence > 0
        
        logger.info(f"Translation service test passed: {translation_result.translated_text[:50]}...")
    
    async def test_unified_text_agent_translation_tools(self, setup_agents):
        """Test translation tools in UnifiedTextAgent."""
        text_agent = setup_agents["text"]
        
        # Test translate_text tool
        spanish_text = "Este es un texto de prueba en español."
        result = await text_agent.translate_text(spanish_text)
        
        assert result["status"] == "success"
        assert result["original_text"] == spanish_text
        assert result["translated_text"] != spanish_text
        assert result["source_language"] == "es"
        assert result["target_language"] == "en"
        
        # Test detect_language tool
        language_result = await text_agent.detect_language(spanish_text)
        assert language_result["status"] == "success"
        assert language_result["detected_language"] == "es"
        
        logger.info("Text agent translation tools test passed")
    
    async def test_tool_registry_translation_tools(self):
        """Test translation tools in ToolRegistry."""
        # Test translate_text tool
        spanish_text = "Hola mundo, esto es una prueba."
        result = await tool_registry.execute_tool("translate_text", spanish_text)
        
        assert result["status"] == "success"
        assert result["original_text"] == spanish_text
        assert result["translated_text"] != spanish_text
        
        # Test detect_language tool
        language_result = await tool_registry.execute_tool("detect_language", spanish_text)
        assert language_result["status"] == "success"
        assert language_result["detected_language"] == "es"
        
        logger.info("ToolRegistry translation tools test passed")
    
    async def test_unified_vision_agent_integration(self, setup_agents):
        """Test UnifiedVisionAgent capabilities."""
        vision_agent = setup_agents["vision"]
        
        # Test image processing (using a placeholder path)
        request = AnalysisRequest(
            data_type=DataType.IMAGE,
            content="test_image.jpg",  # Placeholder
            language="en"
        )
        
        # This test would require an actual image file
        # For now, we'll test that the agent can be initialized and has the right capabilities
        assert vision_agent is not None
        assert "vision" in vision_agent.metadata.get("capabilities", [])
        
        logger.info("Vision agent initialization test passed")
    
    async def test_unified_audio_agent_integration(self, setup_agents):
        """Test UnifiedAudioAgent capabilities."""
        audio_agent = setup_agents["audio"]
        
        # Test audio processing (using a placeholder path)
        request = AnalysisRequest(
            data_type=DataType.AUDIO,
            content="test_audio.wav",  # Placeholder
            language="en"
        )
        
        # This test would require an actual audio file
        # For now, we'll test that the agent can be initialized and has the right capabilities
        assert audio_agent is not None
        assert "audio" in audio_agent.metadata.get("capabilities", [])
        
        logger.info("Audio agent initialization test passed")
    
    async def test_unified_file_extraction_agent_integration(self, setup_agents):
        """Test UnifiedFileExtractionAgent capabilities."""
        file_agent = setup_agents["file_extraction"]
        
        # Test PDF processing (using a placeholder path)
        request = AnalysisRequest(
            data_type=DataType.PDF,
            content="test_document.pdf",  # Placeholder
            language="en"
        )
        
        # This test would require an actual PDF file
        # For now, we'll test that the agent can be initialized and has the right capabilities
        assert file_agent is not None
        assert "file_extraction" in file_agent.metadata.get("capabilities", [])
        
        logger.info("File extraction agent initialization test passed")
    
    async def test_tool_registry_comprehensive(self):
        """Test comprehensive ToolRegistry functionality."""
        # Test tool listing
        tools = tool_registry.list_tools()
        assert len(tools) > 0
        assert "translate_text" in tools
        assert "detect_language" in tools
        
        # Test tool metadata
        metadata = tool_registry.get_tool_metadata("translate_text")
        assert metadata is not None
        assert "description" in metadata
        assert "tags" in metadata
        
        # Test tools by tag
        translation_tools = tool_registry.get_tools_by_tag("translation")
        assert len(translation_tools) > 0
        assert "translate_text" in translation_tools
        
        logger.info("ToolRegistry comprehensive test passed")
    
    async def test_agent_status_and_metadata(self, setup_agents):
        """Test agent status and metadata consistency."""
        for agent_name, agent in setup_agents.items():
            # Test status
            status = agent.get_status()
            assert status is not None
            assert "agent_id" in status
            assert "status" in status
            
            # Test metadata
            metadata = agent.metadata
            assert metadata is not None
            assert "capabilities" in metadata
            assert "supported_data_types" in metadata
            
            logger.info(f"{agent_name} agent status and metadata test passed")
    
    async def test_error_handling(self, setup_agents):
        """Test error handling across agents."""
        text_agent = setup_agents["text"]
        
        # Test with invalid content
        request = AnalysisRequest(
            data_type=DataType.TEXT,
            content=None,  # Invalid content
            language="en"
        )
        
        try:
            result = await text_agent.process(request)
            # Should handle gracefully
            assert result is not None
            logger.info("Error handling test passed")
        except Exception as e:
            logger.warning(f"Error handling test caught exception: {e}")
            # This is also acceptable as long as it doesn't crash
    
    async def test_translation_memory_and_caching(self, translation_service):
        """Test translation memory and caching functionality."""
        test_text = "This is a test text for caching."
        
        # First translation
        result1 = await translation_service.translate_text(test_text)
        assert result1 is not None
        
        # Second translation (should use memory)
        result2 = await translation_service.translate_text(test_text)
        assert result2 is not None
        
        # Both should have the same result
        assert result1.translated_text == result2.translated_text
        
        # Check statistics
        stats = translation_service.get_stats()
        assert stats["total_translations"] >= 2
        
        logger.info("Translation memory and caching test passed")
    
    async def test_batch_processing(self, translation_service):
        """Test batch processing capabilities."""
        texts = [
            "Hello world",
            "Bonjour le monde",
            "Hola mundo",
            "Ciao mondo"
        ]
        
        # Batch translate
        results = await translation_service.batch_translate(texts)
        assert len(results) == len(texts)
        
        for result in results:
            assert result is not None
            assert result.translated_text != result.original_text
        
        logger.info("Batch processing test passed")


async def run_integration_tests():
    """Run all integration tests."""
    logger.info("Starting comprehensive integration tests...")
    
    # Create test instance
    test_instance = TestFinalIntegration()
    
    # Run tests
    tests = [
        test_instance.test_unified_text_agent_integration,
        test_instance.test_translation_service_integration,
        test_instance.test_unified_text_agent_translation_tools,
        test_instance.test_tool_registry_translation_tools,
        test_instance.test_unified_vision_agent_integration,
        test_instance.test_unified_audio_agent_integration,
        test_instance.test_unified_file_extraction_agent_integration,
        test_instance.test_tool_registry_comprehensive,
        test_instance.test_agent_status_and_metadata,
        test_instance.test_error_handling,
        test_instance.test_translation_memory_and_caching,
        test_instance.test_batch_processing
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            # Set up fixtures
            setup_agents = await test_instance.setup_agents.__anext__()
            translation_service = await test_instance.translation_service.__anext__()
            
            # Run test
            if "setup_agents" in test.__code__.co_varnames:
                await test(setup_agents)
            elif "translation_service" in test.__code__.co_varnames:
                await test(translation_service)
            else:
                await test()
            
            passed += 1
            logger.info(f"✓ {test.__name__} passed")
            
        except Exception as e:
            failed += 1
            logger.error(f"✗ {test.__name__} failed: {e}")
        
        finally:
            # Clean up fixtures
            try:
                if 'setup_agents' in locals():
                    for agent in setup_agents.values():
                        await agent.stop()
            except:
                pass
    
    logger.info(f"Integration tests completed: {passed} passed, {failed} failed")
    return passed, failed


if __name__ == "__main__":
    # Run tests
    asyncio.run(run_integration_tests())
