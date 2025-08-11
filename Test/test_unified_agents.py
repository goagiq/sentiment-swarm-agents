#!/usr/bin/env python3
"""
Comprehensive test suite for unified agents.
Tests all functionality of UnifiedTextAgent, UnifiedAudioAgent, and UnifiedVisionAgent.
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agents.unified_text_agent import UnifiedTextAgent
from agents.unified_audio_agent import UnifiedAudioAgent
from agents.unified_vision_agent import UnifiedVisionAgent
from core.models import AnalysisRequest, DataType


class TestUnifiedAgents:
    """Test suite for unified agents."""
    
    def setup_method(self):
        """Set up test environment."""
        self.text_agent = UnifiedTextAgent()
        self.audio_agent = UnifiedAudioAgent()
        self.vision_agent = UnifiedVisionAgent()
        
        # Test data
        self.test_text = "I love this product! It's amazing and works perfectly."
        self.test_text_negative = "This is terrible. I hate it and it doesn't work at all."
        
    async def test_text_agent_initialization(self):
        """Test UnifiedTextAgent initialization."""
        assert self.text_agent is not None
        assert hasattr(self.text_agent, 'process')
        assert hasattr(self.text_agent, 'can_process')
        assert 'text' in self.text_agent.metadata['capabilities']
        
    async def test_text_agent_simple_mode(self):
        """Test UnifiedTextAgent in simple mode."""
        agent = UnifiedTextAgent(use_strands=False, use_swarm=False)
        request = AnalysisRequest(
            content=self.test_text,
            data_type=DataType.TEXT,
            language="en"
        )
        
        result = await agent.process(request)
        assert result is not None
        assert result.sentiment is not None
        assert result.sentiment.label in ['positive', 'negative', 'neutral']
        
    async def test_text_agent_strands_mode(self):
        """Test UnifiedTextAgent in strands mode."""
        agent = UnifiedTextAgent(use_strands=True, use_swarm=False)
        request = AnalysisRequest(
            content=self.test_text,
            data_type=DataType.TEXT,
            language="en"
        )
        
        result = await agent.process(request)
        assert result is not None
        assert result.sentiment is not None
        
    async def test_text_agent_swarm_mode(self):
        """Test UnifiedTextAgent in swarm mode."""
        agent = UnifiedTextAgent(use_strands=True, use_swarm=True, agent_count=2)
        request = AnalysisRequest(
            content=self.test_text,
            data_type=DataType.TEXT,
            language="en"
        )
        
        result = await agent.process(request)
        assert result is not None
        assert result.sentiment is not None
        
    async def test_text_agent_sentiment_analysis(self):
        """Test text sentiment analysis."""
        request = AnalysisRequest(
            content=self.test_text,
            data_type=DataType.TEXT,
            language="en"
        )
        
        result = await self.text_agent.process(request)
        assert result.sentiment.label == 'positive'
        assert result.sentiment.confidence > 0.5
        
        # Test negative sentiment
        request_negative = AnalysisRequest(
            content=self.test_text_negative,
            data_type=DataType.TEXT,
            language="en"
        )
        
        result_negative = await self.text_agent.process(request_negative)
        assert result_negative.sentiment.label == 'negative'
        
    async def test_audio_agent_initialization(self):
        """Test UnifiedAudioAgent initialization."""
        assert self.audio_agent is not None
        assert hasattr(self.audio_agent, 'process')
        assert hasattr(self.audio_agent, 'can_process')
        assert 'audio' in self.audio_agent.metadata['capabilities']
        
    async def test_audio_agent_capabilities(self):
        """Test UnifiedAudioAgent capabilities."""
        agent = UnifiedAudioAgent(enable_summarization=True, enable_large_file_processing=True)
        capabilities = agent.metadata['capabilities']
        
        assert 'audio' in capabilities
        assert 'transcription' in capabilities
        assert 'sentiment_analysis' in capabilities
        assert 'audio_summarization' in capabilities
        assert 'large_file_processing' in capabilities
        
    async def test_vision_agent_initialization(self):
        """Test UnifiedVisionAgent initialization."""
        assert self.vision_agent is not None
        assert hasattr(self.vision_agent, 'process')
        assert hasattr(self.vision_agent, 'can_process')
        assert 'vision' in self.vision_agent.metadata['capabilities']
        
    async def test_vision_agent_capabilities(self):
        """Test UnifiedVisionAgent capabilities."""
        capabilities = self.vision_agent.metadata['capabilities']
        
        assert 'vision' in capabilities
        assert 'image_analysis' in capabilities
        assert 'sentiment_analysis' in capabilities
        
    async def test_agent_status(self):
        """Test agent status methods."""
        # Test text agent status
        text_status = self.text_agent.get_status()
        assert 'agent_id' in text_status
        assert 'status' in text_status
        assert 'capabilities' in text_status
        
        # Test audio agent status
        audio_status = self.audio_agent.get_status()
        assert 'agent_id' in audio_status
        assert 'status' in audio_status
        assert 'capabilities' in audio_status
        
        # Test vision agent status
        vision_status = self.vision_agent.get_status()
        assert 'agent_id' in vision_status
        assert 'status' in vision_status
        assert 'capabilities' in vision_status
        
    async def test_agent_lifecycle(self):
        """Test agent start/stop lifecycle."""
        # Test text agent
        await self.text_agent.start()
        await self.text_agent.stop()
        
        # Test audio agent
        await self.audio_agent.start()
        await self.audio_agent.stop()
        
        # Test vision agent
        await self.vision_agent.start()
        await self.vision_agent.stop()


async def run_tests():
    """Run all tests."""
    test_suite = TestUnifiedAgents()
    
    # Run all test methods
    test_methods = [method for method in dir(test_suite) if method.startswith('test_')]
    
    print(f"Running {len(test_methods)} tests...")
    
    for method_name in test_methods:
        method = getattr(test_suite, method_name)
        if asyncio.iscoroutinefunction(method):
            try:
                await method()
                print(f"✅ {method_name} passed")
            except Exception as e:
                print(f"❌ {method_name} failed: {e}")
        else:
            try:
                method()
                print(f"✅ {method_name} passed")
            except Exception as e:
                print(f"❌ {method_name} failed: {e}")
    
    print("Test suite completed!")


if __name__ == "__main__":
    asyncio.run(run_tests())
