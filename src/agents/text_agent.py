"""
Text processing agent for sentiment analysis using Strands tools.
"""

import asyncio
from typing import Any, Optional

from loguru import logger
from src.core.strands_mock import tool

from agents.base_agent import BaseAgent
from config.config import config
from core.models import (
    AnalysisRequest, 
    AnalysisResult, 
    DataType, 
    SentimentResult
)


class TextAgent(BaseAgent):
    """Agent for processing text-based content using Strands tools."""
    
    def __init__(
        self, 
        model_name: Optional[str] = None, 
        **kwargs
    ):
        # Use config system instead of hardcoded values
        default_model = config.model.default_text_model
        super().__init__(
            model_name=model_name or default_model, 
            **kwargs
        )
        self.metadata["model"] = (
            model_name or default_model
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
        """Process text analysis request using Strands tools."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Extract text content
            text_content = self._extract_text(request.content)
            
            # Use Strands agent to process the request with tool coordination
            system_prompt = (
                "You are a text sentiment analysis expert. Use the available "
                "tools to analyze the sentiment of the given text.\n\n"
                "Available tools:\n"
                "- analyze_text_sentiment: Analyze sentiment using Ollama\n"
                "- extract_text_features: Extract text features for analysis\n"
                "- fallback_sentiment_analysis: Rule-based fallback analysis\n\n"
                "Process the text step by step:\n"
                "1. First extract text features to understand the content\n"
                "2. Then analyze sentiment using the sentiment analysis tool\n"
                "3. If sentiment analysis fails, use the fallback method\n\n"
                "Always use the tools rather than trying to analyze directly."
            )

            # Update the agent's system prompt for this specific task
            self.strands_agent.system_prompt = system_prompt
            
            # Invoke the Strands agent with the text analysis request
            prompt = (
                f"Analyze the sentiment of this text: {text_content}\n\n"
                f"Please use the available tools to perform a comprehensive "
                f"analysis."
            )
            response = await self.strands_agent.invoke_async(prompt)
            
            # Parse the response and create sentiment result
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
                    "model": self.metadata["model"],
                    "language": request.language,
                    "method": "strands_tools",
                    "tools_used": [tool.__name__ for tool in self._get_tools()]
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
    
    @tool
    async def analyze_text_sentiment(self, text: str) -> dict:
        """Analyze the sentiment of the given text using Ollama."""
        try:
            # Simple sentiment analysis using Ollama
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": "phi3:mini",
                    "prompt": f"""Analyze the sentiment of this text and respond with exactly one word: POSITIVE, NEGATIVE, or NEUTRAL.

Text: {text}

Sentiment:""",
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 10,
                        "top_k": 1,
                        "top_p": 0.1
                    }
                }
                
                async with session.post(
                    "http://localhost:11434/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=3)
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
                            positive_score = 0.1
                            negative_score = 0.8
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
            # Fallback to rule-based sentiment analysis
            return await self.fallback_sentiment_analysis(text)
    
    @tool
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
    
    @tool
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
    
    def _parse_sentiment_response(self, response: str) -> SentimentResult:
        """Parse the sentiment response from Strands tools."""
        try:
            # Try to extract sentiment information from the response
            response_lower = response.lower()
            
            if "positive" in response_lower:
                label = "positive"
                confidence = 0.8
            elif "negative" in response_lower:
                label = "negative"
                confidence = 0.8
            else:
                label = "neutral"
                confidence = 0.6
            
            return SentimentResult(
                label=label,
                confidence=confidence,
                scores={
                    "positive": 0.8 if label == "positive" else 0.1,
                    "negative": 0.8 if label == "negative" else 0.1,
                    "neutral": 0.6 if label == "neutral" else 0.1
                },
                metadata={
                    "method": "strands_tools",
                    "raw_response": response,
                    "agent_tools_used": True
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to parse sentiment response: {e}")
            # Return neutral sentiment on parsing failure
            return SentimentResult(
                label="neutral",
                confidence=0.0,
                scores={"neutral": 1.0},
                metadata={
                    "method": "strands_tools",
                    "error": str(e),
                    "raw_response": response
                }
            )
