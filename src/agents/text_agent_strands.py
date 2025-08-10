"""
Text processing agent for sentiment analysis using Strands framework.
"""

import asyncio
from typing import Any, Optional

from loguru import logger
from src.config.config import config
from src.core.models import (
    AnalysisRequest, AnalysisResult, DataType, SentimentResult
)
from src.core.ollama_integration import create_ollama_agent


async def analyze_text_sentiment(text: str) -> str:
    """
    Analyze the sentiment of the given text using Ollama.
    
    Args:
        text: The text to analyze for sentiment
        
    Returns:
        A sentiment analysis result in JSON format
    """
    try:
        # Create a specialized sentiment analysis agent with Ollama
        sentiment_agent = create_ollama_agent(
            model_type="text",
            name="sentiment_analyzer",
            system_prompt=(
                "You are a specialized sentiment analysis expert. "
                "Analyze the given text and respond with exactly one word: "
                "POSITIVE, NEGATIVE, or NEUTRAL. "
                "Then provide a confidence score from 0.0 to 1.0. "
                "Format your response as: "
                "SENTIMENT: [POSITIVE/NEGATIVE/NEUTRAL], "
                "CONFIDENCE: [0.0-1.0]"
            )
        )
        
        if not sentiment_agent:
            logger.warning("Ollama agent not available, using fallback")
            return await fallback_sentiment_analysis(text)
        
        # Get the response
        response = await sentiment_agent.invoke_async(
            f"Analyze this text: {text}"
        )
        return str(response)
        
    except Exception as e:
        logger.error(f"Ollama sentiment analysis failed: {e}")
        return await fallback_sentiment_analysis(text)


async def fallback_sentiment_analysis(text: str) -> str:
    """
    Fallback rule-based sentiment analysis when Strands fails.
    
    Args:
        text: The text to analyze
        
    Returns:
        A sentiment analysis result in JSON format
    """
    text_lower = text.lower()
    
    # Simple keyword-based sentiment analysis
    positive_words = {
        'love', 'great', 'good', 'excellent', 'amazing', 'wonderful', 
        'fantastic', 'awesome', 'brilliant', 'outstanding', 'perfect',
        'happy', 'joy', 'pleased', 'satisfied', 'delighted', 'thrilled'
    }
    
    negative_words = {
        'hate', 'terrible', 'awful', 'horrible', 'bad', 'worst',
        'disappointed', 'angry', 'sad', 'upset', 'frustrated',
        'annoyed', 'disgusted', 'furious', 'miserable'
    }
    
    # Count positive and negative words
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)
    
    # Determine sentiment
    if pos_count > neg_count:
        sentiment_label = "POSITIVE"
        confidence = min(0.7 + (pos_count * 0.1), 0.95)
    elif neg_count > pos_count:
        sentiment_label = "NEGATIVE"
        confidence = min(0.7 + (neg_count * 0.1), 0.95)
    else:
        sentiment_label = "NEUTRAL"
        confidence = 0.6
    
    return f"SENTIMENT: {sentiment_label}, CONFIDENCE: {confidence:.2f}"


class TextAgentStrands:
    """Agent for processing text-based content using Strands framework."""
    
    def __init__(self, model_name: Optional[str] = None):
        # Use config system instead of hardcoded values
        default_model = config.model.default_text_model
        self.model_name = model_name or default_model
        self.agent_id = f"TextAgentStrands_{id(self)}"
        
        # Create the main agent with Ollama integration
        self.agent = create_ollama_agent(
            model_type="text",
            name="text_analyzer",
            system_prompt=(
                "You are a text sentiment analysis agent. "
                "Use the available analysis functions to analyze text "
                "sentiment efficiently. Always try the Ollama-based "
                "sentiment analysis first, then fallback to rule-based "
                "if needed."
            )
        )
        
        if not self.agent:
            logger.warning("Ollama agent not available, using fallback")
        
        logger.info(f"Initialized TextAgentStrands {self.agent_id}")
    
    async def can_process(self, request: AnalysisRequest) -> bool:
        """Check if this agent can process the request."""
        return request.data_type in [
            DataType.TEXT, 
            DataType.SOCIAL_MEDIA
        ]
    
    async def process(self, request: AnalysisRequest) -> AnalysisResult:
        """Process text analysis request using Ollama."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Extract text content
            text_content = self._extract_text(request.content)
            
            # Use Ollama agent to analyze sentiment
            if self.agent:
                response = await self.agent.invoke_async(
                    f"Analyze the sentiment of this text: {text_content}"
                )
            else:
                # Fallback to direct analysis
                response = await analyze_text_sentiment(text_content)
            
            # Parse the response
            sentiment_result = self._parse_sentiment_response(str(response))
            
            # Create analysis result
            result = AnalysisResult(
                request_id=request.id,
                data_type=request.data_type,
                sentiment=sentiment_result,
                processing_time=0.0,  # Will be set by base class
                status=None,  # Will be set by base class
                raw_content=str(request.content),
                extracted_text=text_content,
                metadata={
                    "agent_id": self.agent_id,
                    "model": self.model_name,
                    "language": request.language,
                    "method": "ollama"
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Text processing failed: {e}")
            # Return neutral sentiment on error
            return AnalysisResult(
                request_id=request.id,
                data_type=request.data_type,
                sentiment=SentimentResult(
                    label="neutral",
                    confidence=0.0,
                    scores={},
                    metadata={"error": str(e)}
                ),
                processing_time=asyncio.get_event_loop().time() - start_time,
                status="failed",
                raw_content=str(request.content),
                extracted_text="",
                metadata={
                    "agent_id": self.agent_id,
                    "model": self.model_name,
                    "language": request.language,
                    "error": str(e)
                }
            )
    
    def _extract_text(self, content: Any) -> str:
        """Extract text content from various input formats."""
        if isinstance(content, str):
            return content
        elif isinstance(content, dict):
            # Handle social media posts, API responses, etc.
            if "text" in content:
                return content["text"]
            elif "content" in content:
                return content["content"]
            elif "message" in content:
                return content["message"]
            else:
                return str(content)
        else:
            return str(content)
    
    def _parse_sentiment_response(self, response: str) -> SentimentResult:
        """Parse the sentiment response from Ollama agent."""
        try:
            response_upper = response.upper()
            
            # Extract sentiment and confidence
            if "SENTIMENT:" in response_upper and "CONFIDENCE:" in response_upper:
                # Parse structured response
                parts = response_upper.split(",")
                sentiment_part = parts[0].strip()
                confidence_part = parts[1].strip()
                
                # Extract sentiment label
                if "POSITIVE" in sentiment_part:
                    sentiment_label = "positive"
                elif "NEGATIVE" in sentiment_part:
                    sentiment_label = "negative"
                else:
                    sentiment_label = "neutral"
                
                # Extract confidence
                try:
                    confidence_str = confidence_part.split(":")[1].strip()
                    confidence = float(confidence_str)
                except (IndexError, ValueError):
                    confidence = 0.6
                    
            else:
                # Fallback parsing
                if "POSITIVE" in response_upper:
                    sentiment_label = "positive"
                    confidence = 0.8
                elif "NEGATIVE" in response_upper:
                    sentiment_label = "negative"
                    confidence = 0.8
                else:
                    sentiment_label = "neutral"
                    confidence = 0.6
            
            return SentimentResult(
                label=sentiment_label,
                confidence=confidence,
                scores={
                    "ollama_confidence": confidence
                },
                reasoning=response,
                metadata={
                    "method": "ollama_agent"
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to parse sentiment response: {e}")
            return SentimentResult(
                label="neutral",
                confidence=0.0,
                scores={},
                metadata={"error": f"Parse error: {e}"}
            )
