"""
Simplified Text processing agent for sentiment analysis without Strands framework.
"""

import asyncio
from typing import Any, Optional

from loguru import logger

from src.agents.base_agent import BaseAgent
from src.config.config import config
from src.core.models import (
    AnalysisRequest, 
    AnalysisResult, 
    DataType, 
    SentimentResult
)


class SimpleTextAgent(BaseAgent):
    """Simplified agent for processing text-based content without Strands."""
    
    def __init__(
        self, 
        model_name: Optional[str] = None, 
        **kwargs
    ):
        super().__init__(
            model_name=model_name or config.model.default_text_model, 
            **kwargs
        )
        self.metadata["model"] = (
            model_name or config.model.default_text_model
        )
        self.metadata["supported_languages"] = ["en"]
        self.metadata["capabilities"] = ["text", "sentiment_analysis"]
    
    def _get_tools(self) -> list:
        """Get list of tools for this agent."""
        return [
            self.analyze_text_sentiment,
            self.extract_text_features,
            self.fallback_sentiment_analysis
        ]
    
    async def can_process(self, request: AnalysisRequest) -> bool:
        """Check if this agent can process the request."""
        return request.data_type in [
            DataType.TEXT, 
            DataType.SOCIAL_MEDIA
        ]
    
    async def process(self, request: AnalysisRequest) -> AnalysisResult:
        """Process text analysis request directly using tools."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Extract text content
            text_content = self._extract_text(request.content)
            
            # First extract text features
            features_result = await self.extract_text_features(text_content)
            
            # Then analyze sentiment using the sentiment analysis tool
            sentiment_result = await self.analyze_text_sentiment(text_content)
            
            # Parse the sentiment response
            sentiment_data = self._parse_sentiment_tool_response(sentiment_result)
            
            # Create analysis result
            result = AnalysisResult(
                request_id=request.id,
                data_type=request.data_type,
                sentiment=sentiment_data,
                processing_time=0.0,  # Will be set by base class
                status=None,  # Will be set by base class
                raw_content=str(request.content),
                extracted_text=text_content,
                metadata={
                    "agent_id": self.agent_id,
                    "model": self.metadata["model"],
                    "language": request.language,
                    "method": "direct_tools",
                    "tools_used": ["extract_text_features", "analyze_text_sentiment"],
                    "features": features_result.get("content", [{}])[0].get("json", {})
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
                    metadata={"error": str(e)}
                ),
                processing_time=asyncio.get_event_loop().time() - start_time,
                status="failed",
                raw_content=str(request.content),
                extracted_text="",
                metadata={
                    "agent_id": self.agent_id,
                    "model": self.metadata["model"],
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
    
    async def analyze_text_sentiment(self, text: str) -> dict:
        """Analyze the sentiment of the given text using Ollama."""
        try:
            # Simple sentiment analysis using Ollama
            import aiohttp
            
            # Get model configuration from config
            from src.config.config import config
            model_config = config.get_strands_model_config("simple_text")
            
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": model_config["model_id"],
                    "prompt": f"""You are a sentiment analysis expert. Analyze the sentiment of this text and respond with exactly one word: POSITIVE, NEGATIVE, or NEUTRAL.

Text: {text}

Sentiment (one word only):""",
                    "stream": False,
                    "options": {
                        "temperature": model_config["temperature"],
                        "num_predict": model_config["max_tokens"],
                        "top_k": 1,
                        "top_p": 0.1
                    }
                }
                
                async with session.post(
                    "http://localhost:11434/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        response_text = result.get("response", "").strip().upper()
                        
                        # Parse the response and convert to numeric scores
                        if "POSITIVE" in response_text:
                            sentiment_label = "positive"
                            confidence = 0.8
                            positive_score = 0.8
                            negative_score = 0.1
                            neutral_score = 0.1
                        elif "NEGATIVE" in response_text:
                            sentiment_label = "negative"
                            confidence = 0.8
                            negative_score = 0.8
                            positive_score = 0.1
                            neutral_score = 0.1
                        else:
                            sentiment_label = "neutral"
                            confidence = 0.6
                            positive_score = 0.2
                            negative_score = 0.2
                            neutral_score = 0.6
                        
                        return {
                            "status": "success",
                            "content": [{
                                "json": {
                                    "sentiment": sentiment_label,
                                    "confidence": confidence,
                                    "scores": {
                                        "positive": positive_score,
                                        "negative": negative_score,
                                        "neutral": neutral_score
                                    },
                                    "raw_response": response_text,
                                    "method": "ollama_analysis"
                                }
                            }]
                        }
                    else:
                        raise RuntimeError(f"Ollama API error: {response.status}")
                        
        except Exception as e:
            logger.error(f"Ollama sentiment analysis failed: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error details: {str(e)}")
            # Fallback to rule-based sentiment analysis
            return await self.fallback_sentiment_analysis(text)
    
    async def extract_text_features(self, text: str) -> dict:
        """Extract text features for sentiment analysis."""
        try:
            # Extract basic text features
            words = text.split()
            word_count = len(words)
            char_count = len(text)
            avg_word_length = char_count / max(word_count, 1)
            
            # Simple feature extraction
            features = {
                "word_count": word_count,
                "char_count": char_count,
                "avg_word_length": round(avg_word_length, 2),
                "has_question": "?" in text,
                "has_exclamation": "!" in text,
                "text_length_category": "short" if char_count < 100 else "medium" if char_count < 500 else "long",
                "analysis_ready": True
            }
            
            return {
                "status": "success",
                "content": [{"json": features}]
            }
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return {
                "status": "error",
                "content": [{"text": f"Error extracting features: {str(e)}"}]
            }
    
    async def fallback_sentiment_analysis(self, text: str) -> dict:
        """Fallback rule-based sentiment analysis when Ollama fails."""
        try:
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
                sentiment_label = "positive"
                confidence = min(0.7 + (pos_count * 0.1), 0.95)
            elif neg_count > pos_count:
                sentiment_label = "negative"
                confidence = min(0.7 + (neg_count * 0.1), 0.95)
            else:
                sentiment_label = "neutral"
                confidence = 0.6
            
            return {
                "status": "success",
                "content": [{
                    "json": {
                        "sentiment": sentiment_label,
                        "confidence": confidence,
                        "scores": {
                            "positive": float(pos_count) / max(pos_count + neg_count, 1),
                            "negative": float(neg_count) / max(pos_count + neg_count, 1),
                            "neutral": 1.0 - (float(pos_count + neg_count) / max(pos_count + neg_count, 1))
                        },
                        "method": "fallback_rule_based",
                        "positive_words_found": pos_count,
                        "negative_words_found": neg_count
                    }
                }]
            }
            
        except Exception as e:
            logger.error(f"Fallback sentiment analysis failed: {e}")
            return {
                "status": "error",
                "content": [{"text": f"Error in fallback analysis: {str(e)}"}]
            }
    
    def _parse_sentiment_tool_response(self, tool_response: dict) -> SentimentResult:
        """Parse the sentiment response from tool output."""
        try:
            if tool_response.get("status") == "success":
                content = tool_response.get("content", [{}])[0]
                sentiment_data = content.get("json", {})
                
                label = sentiment_data.get("sentiment", "neutral")
                confidence = sentiment_data.get("confidence", 0.6)
                scores = sentiment_data.get("scores", {})
                
                return SentimentResult(
                    label=label,
                    confidence=confidence,
                    scores=scores,
                    metadata={
                        "method": sentiment_data.get("method", "direct_tools"),
                        "raw_response": sentiment_data.get("raw_response", ""),
                        "agent_tools_used": True
                    }
                )
            else:
                # If tool failed, use fallback
                raise RuntimeError("Tool returned error status")
                
        except Exception as e:
            logger.error(f"Failed to parse sentiment tool response: {e}")
            # Return neutral sentiment on parsing failure
            return SentimentResult(
                label="neutral",
                confidence=0.0,
                scores={"neutral": 1.0},
                metadata={
                    "method": "direct_tools",
                    "error": str(e),
                    "tool_response": str(tool_response)
                }
            )
