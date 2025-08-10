"""
Tests for SimpleTextAgent functionality.
"""

import pytest

from src.agents.text_agent_simple import SimpleTextAgent
from src.core.models import AnalysisRequest, DataType


class TestSimpleTextAgent:
    """Test SimpleTextAgent functionality."""
    
    @pytest.fixture
    def simple_text_agent(self):
        """Create a simple text agent for testing."""
        return SimpleTextAgent()
    
    @pytest.fixture
    def text_request(self):
        """Create a text analysis request."""
        return AnalysisRequest(
            data_type=DataType.TEXT,
            content="I love this product! It's amazing.",
            language="en"
        )
    
    @pytest.mark.asyncio
    async def test_can_process_text(self, simple_text_agent, text_request):
        """Test that simple text agent can process text requests."""
        assert await simple_text_agent.can_process(text_request) is True
    
    @pytest.mark.asyncio
    async def test_can_process_social_media(self, simple_text_agent):
        """Test that simple text agent can process social media requests."""
        request = AnalysisRequest(
            data_type=DataType.SOCIAL_MEDIA,
            content={"text": "Great day today!", "platform": "twitter"},
            language="en"
        )
        assert await simple_text_agent.can_process(request) is True
    
    @pytest.mark.asyncio
    async def test_cannot_process_other_types(self, simple_text_agent):
        """Test that simple text agent cannot process non-text requests."""
        request = AnalysisRequest(
            data_type=DataType.IMAGE,
            content="image.jpg",
            language="en"
        )
        assert await simple_text_agent.can_process(request) is False
    
    @pytest.mark.asyncio
    async def test_positive_sentiment_analysis(self, simple_text_agent):
        """Test positive sentiment analysis."""
        request = AnalysisRequest(
            data_type=DataType.TEXT,
            content="I absolutely love this! It's fantastic and wonderful!",
            language="en"
        )
        
        result = await simple_text_agent.process(request)
        assert result.sentiment.label == "positive"
        assert result.sentiment.confidence > 0.7
        assert result.status is None  # Will be set by base class
        assert result.extracted_text == "I absolutely love this! It's fantastic and wonderful!"
    
    @pytest.mark.asyncio
    async def test_negative_sentiment_analysis(self, simple_text_agent):
        """Test negative sentiment analysis."""
        request = AnalysisRequest(
            data_type=DataType.TEXT,
            content="I hate this product. It's terrible and awful.",
            language="en"
        )
        
        result = await simple_text_agent.process(request)
        assert result.sentiment.label == "negative"
        assert result.sentiment.confidence > 0.7
        assert result.extracted_text == "I hate this product. It's terrible and awful."
    
    @pytest.mark.asyncio
    async def test_neutral_sentiment_analysis(self, simple_text_agent):
        """Test neutral sentiment analysis."""
        request = AnalysisRequest(
            data_type=DataType.TEXT,
            content="The product works as expected. It's okay.",
            language="en"
        )
        
        result = await simple_text_agent.process(request)
        assert result.sentiment.label == "neutral"
        assert result.sentiment.confidence > 0.5
        assert result.extracted_text == "The product works as expected. It's okay."
    
    @pytest.mark.asyncio
    async def test_text_feature_extraction(self, simple_text_agent):
        """Test text feature extraction."""
        text = "Hello world! How are you today?"
        features_result = await simple_text_agent.extract_text_features(text)
        
        assert features_result["status"] == "success"
        features = features_result["content"][0]["json"]
        
        assert features["word_count"] == 7
        assert features["char_count"] == 30
        assert features["has_question"] is True
        assert features["has_exclamation"] is True
        assert features["text_length_category"] == "short"
        assert features["analysis_ready"] is True
    
    @pytest.mark.asyncio
    async def test_fallback_sentiment_analysis(self, simple_text_agent):
        """Test fallback sentiment analysis."""
        text = "This is a test message without strong sentiment words."
        fallback_result = await simple_text_agent.fallback_sentiment_analysis(text)
        
        assert fallback_result["status"] == "success"
        sentiment_data = fallback_result["content"][0]["json"]
        
        assert "sentiment" in sentiment_data
        assert "confidence" in sentiment_data
        assert "scores" in sentiment_data
        assert sentiment_data["method"] == "fallback_rule_based"
    
    @pytest.mark.asyncio
    async def test_social_media_processing(self, simple_text_agent):
        """Test social media content processing."""
        social_post = {
            "text": "Amazing experience at the restaurant! 🍕",
            "platform": "instagram",
            "user": "foodie",
            "likes": 128
        }
        
        request = AnalysisRequest(
            data_type=DataType.SOCIAL_MEDIA,
            content=social_post,
            language="en"
        )
        
        result = await simple_text_agent.process(request)
        assert result.extracted_text == "Amazing experience at the restaurant! 🍕"
        assert result.sentiment.label == "positive"
        assert result.metadata["method"] == "direct_tools"
    
    @pytest.mark.asyncio
    async def test_error_handling(self, simple_text_agent):
        """Test error handling in sentiment analysis."""
        # Test with empty content
        request = AnalysisRequest(
            data_type=DataType.TEXT,
            content="",
            language="en"
        )
        
        result = await simple_text_agent.process(request)
        assert result.sentiment.label == "neutral"
        assert result.sentiment.confidence == 0.0
        assert result.status == "failed"
    
    @pytest.mark.asyncio
    async def test_agent_metadata(self, simple_text_agent):
        """Test agent metadata and configuration."""
        assert simple_text_agent.metadata["model"] == "phi3:mini"
        assert simple_text_agent.metadata["supported_languages"] == ["en"]
        assert simple_text_agent.metadata["capabilities"] == ["text", "sentiment_analysis"]
        
        tools = simple_text_agent._get_tools()
        assert len(tools) == 3
        assert "analyze_text_sentiment" in [tool.__name__ for tool in tools]
        assert "extract_text_features" in [tool.__name__ for tool in tools]
        assert "fallback_sentiment_analysis" in [tool.__name__ for tool in tools]


if __name__ == "__main__":
    pytest.main([__file__])
