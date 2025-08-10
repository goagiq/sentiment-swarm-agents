"""
Web scraping agent for sentiment analysis of webpage content using Strands tools.
"""

import asyncio
import re
from typing import Any, Optional
from urllib.parse import urlparse

import aiohttp
from bs4 import BeautifulSoup
from loguru import logger
from agents.base_agent import BaseAgent
from config.config import config
from core.models import AnalysisRequest, AnalysisResult, DataType, SentimentResult
from core.strands_mock import tool


class WebAgent(BaseAgent):
    """Agent for processing webpage content using Strands tools."""
    
    def __init__(self, model_name: Optional[str] = None, **kwargs):
        # Use config system instead of hardcoded values
        default_model = config.model.default_text_model
        super().__init__(model_name=model_name or default_model, **kwargs)
        self.model_name = model_name or default_model
        self.metadata["model"] = self.model_name
        self.metadata["max_content_length"] = 10000  # characters
        self.metadata["timeout"] = 30  # seconds
        self.metadata["user_agent"] = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.124 Safari/537.36"
        )
        self.metadata["capabilities"] = ["web", "scraping", "sentiment_analysis"]
    
    def _get_tools(self) -> list:
        """Get list of tools for this agent."""
        return [
            self.scrape_webpage,
            self.analyze_webpage_sentiment,
            self.extract_webpage_features,
            self.fallback_webpage_analysis
        ]
    
    async def can_process(self, request: AnalysisRequest) -> bool:
        """Check if this agent can process the request."""
        return request.data_type == DataType.WEBPAGE
    
    async def process(self, request: AnalysisRequest) -> AnalysisResult:
        """Process webpage analysis request using Strands tools."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Initialize models if not already done
            await self._initialize_models()
            
            # Extract webpage content
            webpage_content = await self._extract_webpage_content(request.content)
            
            # Use Strands agent to process the request with tool coordination
            system_prompt = (
                "You are a webpage sentiment analysis expert. Use the available "
                "tools to analyze the sentiment of the given webpage content.\n\n"
                "Available tools:\n"
                "- scrape_webpage: Scrape and extract webpage content\n"
                "- analyze_webpage_sentiment: Analyze webpage sentiment\n"
                "- extract_webpage_features: Extract webpage features\n"
                "- fallback_webpage_analysis: Fallback analysis when main tools fail\n\n"
                "Process the webpage content step by step:\n"
                "1. First scrape the webpage to get content\n"
                "2. Then analyze sentiment using the sentiment analysis tool\n"
                "3. If analysis fails, use the fallback method\n\n"
                "Always use the tools rather than trying to analyze directly."
            )
            
            # Update the agent's system prompt for this specific task
            self.strands_agent.system_prompt = system_prompt
            
            # Invoke the Strands agent with the webpage analysis request
            prompt = (
                f"Analyze the sentiment of this webpage: {request.content}\n\n"
                f"Please use the available tools to perform a comprehensive "
                f"analysis."
            )
            response = await self.strands_agent.invoke_async(prompt)
            
            # Parse the response and create sentiment result
            sentiment_result = await self._analyze_webpage_sentiment(webpage_content)
            
            # Create analysis result
            metadata = {
                "agent_id": self.agent_id,
                "model": self.model_name,
                "url": webpage_content.get("url", ""),
                "title": webpage_content.get("title", ""),
                "content_length": len(webpage_content.get("text", "")),
                "method": "strands_tools",
                "tools_used": [tool.__name__ for tool in self._get_tools()]
            }
            
            # Add video platform information if detected
            if "video_platform" in webpage_content:
                metadata.update({
                    "warning": webpage_content.get("content_warning", ""),
                    "platform": webpage_content.get("video_platform", ""),
                    "limitation": webpage_content.get("video_limitation", ""),
                    "suggestions": webpage_content.get("video_suggestions", []),
                    "available_content": webpage_content.get("text", "")[:200] + "..." if len(webpage_content.get("text", "")) > 200 else webpage_content.get("text", "")
                })
            
            result = AnalysisResult(
                request_id=request.id,
                data_type=request.data_type,
                sentiment=sentiment_result,
                processing_time=0.0,  # Will be set by base class
                status=None,  # Will be set by base class
                raw_content=str(request.content),
                extracted_text=webpage_content.get("text", ""),
                metadata=metadata
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Webpage processing failed: {e}")
            # Return neutral sentiment on error
            return AnalysisResult(
                request_id=request.id,
                data_type=request.data_type,
                sentiment=SentimentResult(
                    label="neutral",
                    confidence=0.0,
                    context_notes=f"Error: {str(e)}"
                ),
                processing_time=asyncio.get_event_loop().time() - start_time,
                status="failed",
                raw_content=str(request.content),
                extracted_text="",
                metadata={
                    "agent_id": self.agent_id,
                    "model": self.model_name,
                    "error": str(e),
                    "context_notes": f"Error: {str(e)}"
                }
            )
    
    async def _initialize_models(self):
        """Initialize the models for sentiment analysis."""
        try:
            # No initialization needed for Ollama
            logger.info(f"WebAgent {self.agent_id} initialized with Ollama")
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise
    
    async def _extract_webpage_content(self, content: Any) -> dict:
        """Extract content from webpage."""
        try:
            # Handle different input formats
            if isinstance(content, str):
                url = content
            elif isinstance(content, dict) and "url" in content:
                url = content["url"]
            else:
                raise ValueError("Unsupported webpage content format")
            
            # Validate URL
            if not self._is_valid_url(url):
                raise ValueError(f"Invalid URL: {url}")
            
            # Check if this is a video platform
            video_info = self._is_video_platform(url)
            
            # Fetch webpage content
            webpage_data = await self._fetch_webpage(url)
            
            # Extract and clean text content
            cleaned_text = self._clean_webpage_text(webpage_data["html"])
            
            # Truncate if too long
            if len(cleaned_text) > self.metadata["max_content_length"]:
                cleaned_text = cleaned_text[:self.metadata["max_content_length"]] + "..."
            
            # Add video platform information if detected
            result = {
                "url": url,
                "title": webpage_data["title"],
                "text": cleaned_text,
                "html": webpage_data["html"],
                "status_code": webpage_data["status_code"]
            }
            
            if video_info["is_video"]:
                result.update({
                    "video_platform": video_info["platform"],
                    "video_limitation": video_info["limitation"],
                    "video_suggestions": video_info["suggestions"],
                    "content_warning": (
                        f"⚠️ This is a {video_info['platform']} video. "
                        f"{video_info['limitation']} "
                        "Only page metadata and text content are available for analysis."
                    )
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Webpage content extraction failed: {e}")
            return {
                "error": str(e),
                "text": f"Error extracting webpage content: {str(e)}"
            }
    
    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False
    
    def _is_video_platform(self, url: str) -> dict:
        """Detect if URL is from a video platform and provide guidance."""
        url_lower = url.lower()
        
        # YouTube detection
        if "youtube.com" in url_lower or "youtu.be" in url_lower:
            return {
                "is_video": True,
                "platform": "YouTube",
                "limitation": (
                    "Video content cannot be accessed directly. "
                    "Only page metadata is available."
                ),
                "suggestions": [
                    "Provide the video title and description for text-based analysis",
                    "Share a screenshot of the video for visual analysis",
                    "Use the video transcript if available",
                    "Describe the video content in your own words"
                ]
            }
        
        # Vimeo detection
        elif "vimeo.com" in url_lower:
            return {
                "is_video": True,
                "platform": "Vimeo",
                "limitation": (
                    "Video content cannot be accessed directly. "
                    "Only page metadata is available."
                ),
                "suggestions": [
                    "Provide the video title and description for text-based analysis",
                    "Share a screenshot of the video for visual analysis",
                    "Use the video transcript if available",
                    "Describe the video content in your own words"
                ]
            }
        
        # TikTok detection
        elif "tiktok.com" in url_lower:
            return {
                "is_video": True,
                "platform": "TikTok",
                "limitation": (
                    "Video content cannot be accessed directly. "
                    "Only page metadata is available."
                ),
                "suggestions": [
                    "Provide the video caption and description for text-based analysis",
                    "Share a screenshot of the video for visual analysis",
                    "Describe the video content in your own words"
                ]
            }
        
        # Instagram detection
        elif "instagram.com" in url_lower:
            return {
                "is_video": True,
                "platform": "Instagram",
                "limitation": (
                    "Video content cannot be accessed directly. "
                    "Only page metadata is available."
                ),
                "suggestions": [
                    "Provide the post caption and description for text-based analysis",
                    "Share a screenshot of the video for visual analysis",
                    "Describe the video content in your own words"
                ]
            }
        
        return {
            "is_video": False,
            "platform": None,
            "limitation": None,
            "suggestions": []
        }
    
    async def _fetch_webpage(self, url: str) -> dict:
        """Fetch webpage content using aiohttp."""
        timeout = aiohttp.ClientTimeout(total=self.metadata["timeout"])
        headers = {"User-Agent": self.metadata["user_agent"]}
        
        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
            async with session.get(url) as response:
                html_content = await response.text()
                
                # Extract title
                soup = BeautifulSoup(html_content, "html.parser")
                title = soup.find("title")
                title_text = title.get_text().strip() if title else "No title"
                
                return {
                    "html": html_content,
                    "title": title_text,
                    "status_code": response.status
                }
    
    def _clean_webpage_text(self, html_content: str) -> str:
        """Clean and extract text content from HTML."""
        try:
            soup = BeautifulSoup(html_content, "html.parser")
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "header", "footer"]):
                script.decompose()
            
            # Extract text
            text = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = " ".join(chunk for chunk in chunks if chunk)
            
            # Remove extra whitespace
            text = re.sub(r"\s+", " ", text)
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Text cleaning failed: {e}")
            return f"Error cleaning text: {str(e)}"
    
    async def _analyze_webpage_sentiment(self, webpage_content: dict) -> SentimentResult:
        """Analyze sentiment of webpage content."""
        try:
            if "error" in webpage_content:
                return SentimentResult(
                    label="neutral",
                    confidence=0.0,
                    metadata={"error": webpage_content["error"]}
                )
            
            text_content = webpage_content.get("text", "")
            
            # Check for video platform limitations
            if "video_platform" in webpage_content:
                return SentimentResult(
                    label="neutral",
                    confidence=0.0,
                    metadata={
                        "warning": webpage_content.get("content_warning", ""),
                        "platform": webpage_content.get("video_platform", ""),
                        "limitation": webpage_content.get("video_limitation", ""),
                        "suggestions": webpage_content.get("video_suggestions", []),
                        "available_content": text_content[:200] + "..." if len(text_content) > 200 else text_content
                    }
                )
            
            if not text_content or len(text_content.strip()) < 10:
                return SentimentResult(
                    label="neutral",
                    confidence=0.0,
                    metadata={"error": "Insufficient text content"}
                )
            
            # Run sentiment analysis on the text content
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self.sentiment_pipeline,
                text_content
            )
            
            # Extract sentiment information
            if isinstance(result, list) and len(result) > 0:
                sentiment_data = result[0]
                label = sentiment_data.get("label", "neutral").lower()
                score = sentiment_data.get("score", 0.0)
                
                # Map labels to our enum
                if "pos" in label:
                    sentiment_label = "positive"
                elif "neg" in label:
                    sentiment_label = "negative"
                else:
                    sentiment_label = "neutral"
                
                return SentimentResult(
                    label=sentiment_label,
                    confidence=score,
                    scores={
                        "raw_score": score,
                        "content_length": len(text_content),
                        "title_length": len(webpage_content.get("title", ""))
                    },
                    metadata={
                        "original_label": label,
                        "url": webpage_content.get("url", ""),
                        "title": webpage_content.get("title", ""),
                        "content_preview": text_content[:200] + "..." if len(text_content) > 200 else text_content
                    }
                )
            else:
                return SentimentResult(
                    label="neutral",
                    confidence=0.0,
                    metadata={"error": "No sentiment result"}
                )
                
        except Exception as e:
            logger.error(f"Webpage sentiment analysis failed: {e}")
            return SentimentResult(
                label="neutral",
                confidence=0.0,
                metadata={"error": str(e), "content": webpage_content}
            )

    # Tool methods for Strands integration
    @tool
    async def scrape_webpage(self, url: str) -> dict:
        """Scrape and extract content from a webpage."""
        try:
            if not self._is_valid_url(url):
                return {"error": "Invalid URL format"}
            
            webpage_content = await self._fetch_webpage(url)
            return webpage_content
            
        except Exception as e:
            logger.error(f"Webpage scraping failed: {e}")
            return {"error": str(e)}

    @tool
    async def analyze_webpage_sentiment(self, webpage_content: dict) -> dict:
        """Analyze sentiment of webpage content."""
        try:
            sentiment_result = await self._analyze_webpage_sentiment(webpage_content)
            return {
                "sentiment": sentiment_result.label,
                "confidence": sentiment_result.confidence,
                "metadata": sentiment_result.metadata
            }
        except Exception as e:
            logger.error(f"Webpage sentiment analysis failed: {e}")
            return {"error": str(e)}

    @tool
    async def extract_webpage_features(self, webpage_content: dict) -> dict:
        """Extract features from webpage content."""
        try:
            text_content = webpage_content.get("text", "")
            title = webpage_content.get("title", "")
            url = webpage_content.get("url", "")
            
            features = {
                "text_length": len(text_content),
                "title_length": len(title),
                "has_title": bool(title),
                "url": url,
                "word_count": len(text_content.split()) if text_content else 0,
                "language": "en"  # Default assumption
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return {"error": str(e)}

    @tool
    async def fallback_webpage_analysis(self, webpage_content: dict) -> dict:
        """Fallback analysis when main tools fail."""
        try:
            text_content = webpage_content.get("text", "")
            if not text_content:
                return {"error": "No text content available"}
            
            # Simple rule-based analysis
            positive_words = ["good", "great", "excellent", "amazing", "wonderful", "love", "like"]
            negative_words = ["bad", "terrible", "awful", "hate", "dislike", "horrible", "worst"]
            
            text_lower = text_content.lower()
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)
            
            if positive_count > negative_count:
                sentiment = "positive"
                confidence = min(0.8, positive_count / 10)
            elif negative_count > positive_count:
                sentiment = "negative"
                confidence = min(0.8, negative_count / 10)
            else:
                sentiment = "neutral"
                confidence = 0.5
            
            return {
                "sentiment": sentiment,
                "confidence": confidence,
                "method": "rule_based",
                "positive_words": positive_count,
                "negative_words": negative_count
            }
            
        except Exception as e:
            logger.error(f"Fallback analysis failed: {e}")
            return {"error": str(e)}



