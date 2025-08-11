#!/usr/bin/env python3
"""
Enhanced web scraping agent with yt-dlp integration for YouTube video analysis.
"""

import asyncio
import re
from typing import Any, Optional, Dict
from urllib.parse import urlparse

import aiohttp
import yt_dlp
from bs4 import BeautifulSoup
from loguru import logger
from src.agents.base_agent import BaseAgent
from src.config.config import config
from src.core.models import AnalysisRequest, AnalysisResult, DataType, SentimentResult
from src.core.strands_mock import tool


class EnhancedWebAgent(BaseAgent):
    """Enhanced agent for processing webpage content with yt-dlp integration."""
    
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
        self.metadata["capabilities"] = [
            "web", "scraping", "sentiment_analysis", "yt_dlp_integration"
        ]
    
    def _get_tools(self) -> list:
        """Get list of tools for this agent."""
        return [
            self.scrape_webpage,
            self.analyze_webpage_sentiment,
            self.extract_webpage_features,
            self.fallback_webpage_analysis,
            self.extract_youtube_metadata,
            self.extract_youtube_transcript
        ]
    
    async def can_process(self, request: AnalysisRequest) -> bool:
        """Check if this agent can process the request."""
        return request.data_type == DataType.WEBPAGE
    
    async def process(self, request: AnalysisRequest) -> AnalysisResult:
        """Process webpage analysis request with enhanced capabilities."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Initialize models if not already done
            await self._initialize_models()
            
            # Extract webpage content with enhanced capabilities
            webpage_content = await self._extract_enhanced_content(request.content)
            
            # Create analysis result
            metadata = {
                "agent_id": self.agent_id,
                "model": self.model_name,
                "url": webpage_content.get("url", ""),
                "title": webpage_content.get("title", ""),
                "content_length": len(webpage_content.get("text", "")),
                "method": "enhanced_web_agent",
                "tools_used": [tool.__name__ for tool in self._get_tools()]
            }
            
            # Add enhanced metadata for video platforms
            if "video_platform" in webpage_content:
                metadata.update({
                    "warning": webpage_content.get("content_warning", ""),
                    "platform": webpage_content.get("video_platform", ""),
                    "limitation": webpage_content.get("video_limitation", ""),
                    "suggestions": webpage_content.get("video_suggestions", []),
                    "available_content": webpage_content.get("text", "")[:200] + "..." if len(webpage_content.get("text", "")) > 200 else webpage_content.get("text", ""),
                    "yt_dlp_metadata": webpage_content.get("yt_dlp_metadata", {}),
                    "transcript_available": webpage_content.get("transcript_available", False)
                })
            
            # Analyze sentiment
            sentiment_result = await self._analyze_enhanced_sentiment(webpage_content)
            
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
            logger.error(f"Enhanced webpage processing failed: {e}")
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
            logger.info(f"EnhancedWebAgent {self.agent_id} initialized with Ollama and yt-dlp")
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise
    
    async def _extract_enhanced_content(self, content: Any) -> dict:
        """Extract content from webpage with enhanced capabilities."""
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
            
            # Extract basic webpage content
            webpage_data = await self._fetch_webpage(url)
            cleaned_text = self._clean_webpage_text(webpage_data["html"])
            
            # Truncate if too long
            if len(cleaned_text) > self.metadata["max_content_length"]:
                cleaned_text = cleaned_text[:self.metadata["max_content_length"]] + "..."
            
            # Base result
            result = {
                "url": url,
                "title": webpage_data["title"],
                "text": cleaned_text,
                "html": webpage_data["html"],
                "status_code": webpage_data["status_code"]
            }
            
            # Enhanced processing for video platforms
            if video_info["is_video"]:
                result.update({
                    "video_platform": video_info["platform"],
                    "video_limitation": video_info["limitation"],
                    "video_suggestions": video_info["suggestions"],
                    "content_warning": (
                        f"⚠️ This is a {video_info['platform']} video. "
                        f"{video_info['limitation']} "
                        "Enhanced analysis available with yt-dlp metadata."
                    )
                })
                
                # Try to extract enhanced metadata using yt-dlp
                if video_info["platform"] == "YouTube":
                    try:
                        yt_metadata = await self._extract_youtube_metadata_async(url)
                        if yt_metadata:
                            result["yt_dlp_metadata"] = yt_metadata
                            result["transcript_available"] = yt_metadata.get("has_transcript", False)
                            
                            # Enhance text content with yt-dlp data
                            enhanced_text = self._create_enhanced_text(yt_metadata, cleaned_text)
                            result["text"] = enhanced_text
                            result["enhanced_analysis"] = True
                    except Exception as e:
                        logger.warning(f"yt-dlp extraction failed: {e}")
                        result["yt_dlp_error"] = str(e)
            
            return result
            
        except Exception as e:
            logger.error(f"Enhanced content extraction failed: {e}")
            return {
                "error": str(e),
                "text": f"Error extracting enhanced content: {str(e)}"
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
                    "Enhanced metadata available via yt-dlp."
                ),
                "suggestions": [
                    "Enhanced analysis available with yt-dlp metadata",
                    "Video title and description analysis",
                    "Tags and categories analysis",
                    "Engagement metrics analysis",
                    "Transcript analysis (if available)",
                    "Share a screenshot for visual analysis"
                ]
            }
        
        # Other platforms (same as before)
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
    
    async def _extract_youtube_metadata_async(self, url: str) -> Optional[Dict]:
        """Extract YouTube metadata using enhanced yt-dlp service asynchronously."""
        try:
            # Use the enhanced YouTube-DL service for better reliability
            from src.core.youtube_dl_service import YouTubeDLService
            
            # Create a temporary service instance
            youtube_service = YouTubeDLService(download_path="./temp/youtube_metadata")
            
            # Get metadata using the enhanced service with retry mechanisms
            metadata = await youtube_service.get_metadata(url)
            
            if metadata:
                return {
                    "title": metadata.title,
                    "uploader": "Unknown",  # Not available in VideoMetadata
                    "upload_date": metadata.upload_date,
                    "duration": metadata.duration,
                    "view_count": metadata.view_count,
                    "like_count": metadata.like_count,
                    "comment_count": 0,  # Not available in VideoMetadata
                    "description": metadata.description,
                    "tags": [],  # Not available in VideoMetadata
                    "categories": [],  # Not available in VideoMetadata
                    "has_transcript": False,  # Not available in VideoMetadata
                    "thumbnail_count": 0,  # Not available in VideoMetadata
                    "format_count": len(metadata.available_formats),
                    "enhanced_extraction": True,
                    "platform": metadata.platform
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Enhanced yt-dlp metadata extraction failed: {e}")
            return None
    
    def _create_enhanced_text(self, yt_metadata: Dict, original_text: str) -> str:
        """Create enhanced text content using yt-dlp metadata."""
        enhanced_parts = []
        
        # Add title
        if yt_metadata.get("title"):
            enhanced_parts.append(f"Video Title: {yt_metadata['title']}")
        
        # Add description
        if yt_metadata.get("description"):
            enhanced_parts.append(f"Description: {yt_metadata['description'][:500]}...")
        
        # Add tags
        if yt_metadata.get("tags"):
            tags_text = ", ".join(yt_metadata["tags"][:10])  # Limit to 10 tags
            enhanced_parts.append(f"Tags: {tags_text}")
        
        # Add categories
        if yt_metadata.get("categories"):
            categories_text = ", ".join(yt_metadata["categories"])
            enhanced_parts.append(f"Categories: {categories_text}")
        
        # Add engagement metrics
        engagement_parts = []
        if yt_metadata.get("view_count"):
            engagement_parts.append(f"Views: {yt_metadata['view_count']}")
        if yt_metadata.get("like_count"):
            engagement_parts.append(f"Likes: {yt_metadata['like_count']}")
        if yt_metadata.get("comment_count"):
            engagement_parts.append(f"Comments: {yt_metadata['comment_count']}")
        
        if engagement_parts:
            enhanced_parts.append(f"Engagement: {', '.join(engagement_parts)}")
        
        # Add transcript availability
        if yt_metadata.get("has_transcript"):
            enhanced_parts.append("Transcript: Available for analysis")
        
        # Combine with original text
        enhanced_text = "\n\n".join(enhanced_parts)
        if original_text:
            enhanced_text += f"\n\nOriginal Page Content:\n{original_text}"
        
        return enhanced_text
    
    async def _analyze_enhanced_sentiment(self, webpage_content: dict) -> SentimentResult:
        """Analyze sentiment of enhanced webpage content."""
        try:
            if "error" in webpage_content:
                return SentimentResult(
                    label="neutral",
                    confidence=0.0,
                    context_notes=f"Error: {webpage_content['error']}"
                )
            
            text_content = webpage_content.get("text", "")
            
            # Check for video platform limitations
            if "video_platform" in webpage_content:
                # Enhanced analysis for video platforms
                if webpage_content.get("enhanced_analysis"):
                    return SentimentResult(
                        label="neutral",  # Will be analyzed by text agent
                        confidence=0.8,  # Higher confidence with enhanced data
                        context_notes="Enhanced analysis with yt-dlp metadata available"
                    )
                else:
                    return SentimentResult(
                        label="neutral",
                        confidence=0.0,
                        context_notes=webpage_content.get("content_warning", "")
                    )
            
            if not text_content or len(text_content.strip()) < 10:
                return SentimentResult(
                    label="neutral",
                    confidence=0.0,
                    context_notes="Insufficient text content"
                )
            
            # For regular webpages, return neutral for text agent to analyze
            return SentimentResult(
                label="neutral",
                confidence=0.6,
                context_notes="Content available for text-based sentiment analysis"
            )
                
        except Exception as e:
            logger.error(f"Enhanced sentiment analysis failed: {e}")
            return SentimentResult(
                label="neutral",
                confidence=0.0,
                context_notes=f"Error: {str(e)}"
            )
    
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
    async def extract_youtube_metadata(self, url: str) -> dict:
        """Extract YouTube video metadata using yt-dlp."""
        try:
            if not self._is_valid_url(url):
                return {"error": "Invalid URL format"}
            
            metadata = await self._extract_youtube_metadata_async(url)
            if metadata:
                return {"success": True, "metadata": metadata}
            else:
                return {"error": "Failed to extract metadata"}
                
        except Exception as e:
            logger.error(f"YouTube metadata extraction failed: {e}")
            return {"error": str(e)}
    
    @tool
    async def extract_youtube_transcript(self, url: str) -> dict:
        """Extract YouTube video transcript if available."""
        try:
            if not self._is_valid_url(url):
                return {"error": "Invalid URL format"}
            
            # This would require additional yt-dlp configuration for transcript extraction
            return {"error": "Transcript extraction not yet implemented"}
                
        except Exception as e:
            logger.error(f"YouTube transcript extraction failed: {e}")
            return {"error": str(e)}
    
    @tool
    async def analyze_webpage_sentiment(self, webpage_content: dict) -> dict:
        """Analyze sentiment of webpage content."""
        try:
            sentiment_result = await self._analyze_enhanced_sentiment(webpage_content)
            return {
                "sentiment": sentiment_result.label,
                "confidence": sentiment_result.confidence,
                "context_notes": sentiment_result.context_notes
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
                "language": "en",  # Default assumption
                "enhanced_analysis": webpage_content.get("enhanced_analysis", False),
                "yt_dlp_metadata": bool(webpage_content.get("yt_dlp_metadata"))
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
