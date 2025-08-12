"""
Social Media Agent for integrating with social media platforms and analyzing social media content.
"""

import asyncio
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger

from src.agents.base_agent import StrandsBaseAgent
from src.core.models import AnalysisRequest, AnalysisResult, DataType, SentimentResult
from src.core.error_handler import with_error_handling


class SocialMediaPlatformManager:
    """Manage social media platform integrations."""
    
    def __init__(self):
        self.platforms = {
            "twitter": TwitterPlatform(),
            "linkedin": LinkedInPlatform(),
            "facebook": FacebookPlatform(),
            "instagram": InstagramPlatform()
        }
        self.cache = {}
    
    @with_error_handling("social_media_integration")
    async def integrate_platform_data(
        self,
        platforms: List[str],
        data_types: List[str],
        time_range: str = "7d",
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """Integrate data from multiple social media platforms."""
        try:
            logger.info(f"Integrating data from platforms: {platforms}")
            
            results = {}
            for platform in platforms:
                if platform in self.platforms:
                    platform_data = await self.platforms[platform].fetch_data(
                        data_types, time_range, include_metadata
                    )
                    results[platform] = platform_data
                else:
                    logger.warning(f"Platform {platform} not supported")
                    results[platform] = {"error": "Platform not supported"}
            
            # Combine and analyze data
            combined_data = await self._combine_platform_data(results)
            
            logger.info(f"Social media integration completed for {len(platforms)} platforms")
            return {
                "platforms_processed": len(platforms),
                "data_types": data_types,
                "time_range": time_range,
                "platform_data": results,
                "combined_analysis": combined_data
            }
            
        except Exception as e:
            logger.error(f"Social media integration failed: {e}")
            return {"error": str(e)}
    
    async def _combine_platform_data(self, platform_data: Dict[str, Any]) -> Dict[str, Any]:
        """Combine and analyze data from multiple platforms."""
        # Combine posts, comments, and sentiment data
        all_posts = []
        all_comments = []
        all_sentiment = []
        
        for platform, data in platform_data.items():
            if "error" not in data:
                all_posts.extend(data.get("posts", []))
                all_comments.extend(data.get("comments", []))
                all_sentiment.extend(data.get("sentiment", []))
        
        # Analyze combined data
        analysis = {
            "total_posts": len(all_posts),
            "total_comments": len(all_comments),
            "average_sentiment": self._calculate_average_sentiment(all_sentiment),
            "top_topics": await self._extract_top_topics(all_posts + all_comments),
            "engagement_metrics": self._calculate_engagement_metrics(all_posts, all_comments)
        }
        
        return analysis
    
    def _calculate_average_sentiment(self, sentiment_data: List[Dict[str, Any]]) -> float:
        """Calculate average sentiment score."""
        if not sentiment_data:
            return 0.0
        
        total_score = sum(item.get("score", 0) for item in sentiment_data)
        return total_score / len(sentiment_data)
    
    async def _extract_top_topics(self, content: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract top topics from content."""
        # Simple topic extraction (would integrate with NLP in production)
        topics = {}
        for item in content:
            text = item.get("text", "")
            # Simple keyword extraction
            words = text.lower().split()
            for word in words:
                if len(word) > 3:  # Filter short words
                    topics[word] = topics.get(word, 0) + 1
        
        # Return top 10 topics
        sorted_topics = sorted(topics.items(), key=lambda x: x[1], reverse=True)
        return [{"topic": topic, "frequency": freq} for topic, freq in sorted_topics[:10]]
    
    def _calculate_engagement_metrics(self, posts: List[Dict[str, Any]], comments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate engagement metrics."""
        total_likes = sum(post.get("likes", 0) for post in posts)
        total_shares = sum(post.get("shares", 0) for post in posts)
        total_comments = len(comments)
        
        return {
            "total_likes": total_likes,
            "total_shares": total_shares,
            "total_comments": total_comments,
            "average_engagement_rate": (total_likes + total_shares + total_comments) / max(len(posts), 1)
        }


class TwitterPlatform:
    """Twitter platform integration."""
    
    async def fetch_data(self, data_types: List[str], time_range: str, include_metadata: bool) -> Dict[str, Any]:
        """Fetch data from Twitter."""
        # Mock implementation - would use Twitter API in production
        return {
            "posts": [
                {"id": "1", "text": "Sample tweet", "likes": 10, "shares": 5, "sentiment": 0.8},
                {"id": "2", "text": "Another tweet", "likes": 15, "shares": 3, "sentiment": 0.6}
            ],
            "comments": [
                {"id": "1", "text": "Great post!", "sentiment": 0.9},
                {"id": "2", "text": "Interesting", "sentiment": 0.7}
            ],
            "sentiment": [
                {"score": 0.8, "label": "positive"},
                {"score": 0.6, "label": "positive"}
            ],
            "trends": ["#AI", "#Technology", "#Innovation"]
        }


class LinkedInPlatform:
    """LinkedIn platform integration."""
    
    async def fetch_data(self, data_types: List[str], time_range: str, include_metadata: bool) -> Dict[str, Any]:
        """Fetch data from LinkedIn."""
        # Mock implementation - would use LinkedIn API in production
        return {
            "posts": [
                {"id": "1", "text": "Professional update", "likes": 25, "shares": 8, "sentiment": 0.7},
                {"id": "2", "text": "Industry insights", "likes": 30, "shares": 12, "sentiment": 0.8}
            ],
            "comments": [
                {"id": "1", "text": "Great insights!", "sentiment": 0.8},
                {"id": "2", "text": "Very helpful", "sentiment": 0.9}
            ],
            "sentiment": [
                {"score": 0.7, "label": "positive"},
                {"score": 0.8, "label": "positive"}
            ],
            "trends": ["#Leadership", "#Innovation", "#Business"]
        }


class FacebookPlatform:
    """Facebook platform integration."""
    
    async def fetch_data(self, data_types: List[str], time_range: str, include_metadata: bool) -> Dict[str, Any]:
        """Fetch data from Facebook."""
        # Mock implementation - would use Facebook API in production
        return {
            "posts": [
                {"id": "1", "text": "Facebook post", "likes": 50, "shares": 15, "sentiment": 0.6},
                {"id": "2", "text": "Another post", "likes": 35, "shares": 8, "sentiment": 0.7}
            ],
            "comments": [
                {"id": "1", "text": "Nice post!", "sentiment": 0.8},
                {"id": "2", "text": "Thanks for sharing", "sentiment": 0.7}
            ],
            "sentiment": [
                {"score": 0.6, "label": "positive"},
                {"score": 0.7, "label": "positive"}
            ],
            "trends": ["#Social", "#Community", "#Sharing"]
        }


class InstagramPlatform:
    """Instagram platform integration."""
    
    async def fetch_data(self, data_types: List[str], time_range: str, include_metadata: bool) -> Dict[str, Any]:
        """Fetch data from Instagram."""
        # Mock implementation - would use Instagram API in production
        return {
            "posts": [
                {"id": "1", "text": "Instagram post", "likes": 100, "shares": 20, "sentiment": 0.8},
                {"id": "2", "text": "Another post", "likes": 75, "shares": 15, "sentiment": 0.7}
            ],
            "comments": [
                {"id": "1", "text": "Beautiful!", "sentiment": 0.9},
                {"id": "2", "text": "Amazing", "sentiment": 0.8}
            ],
            "sentiment": [
                {"score": 0.8, "label": "positive"},
                {"score": 0.7, "label": "positive"}
            ],
            "trends": ["#Photography", "#Art", "#Creativity"]
        }


class SocialMediaAgent(StrandsBaseAgent):
    """
    Social Media Agent for integrating with social media platforms and analyzing content.
    
    Supports:
    - Multiple social media platform integrations
    - Social media content analysis
    - Trend monitoring and analysis
    - Engagement metrics calculation
    """
    
    def __init__(self, model_name: Optional[str] = None, **kwargs):
        super().__init__(model_name=model_name or "mistral-small3.1:latest", **kwargs)
        
        # Initialize social media components
        self.platform_manager = SocialMediaPlatformManager()
        
        # Set metadata
        self.metadata["agent_type"] = "social_media"
        self.metadata["capabilities"] = [
            "platform_integration",
            "content_analysis",
            "trend_monitoring",
            "engagement_metrics",
            "sentiment_analysis"
        ]
        self.metadata["supported_platforms"] = ["twitter", "linkedin", "facebook", "instagram"]
        
        logger.info("SocialMediaAgent initialized successfully")
    
    async def can_process(self, request: AnalysisRequest) -> bool:
        """Check if this agent can process the given request."""
        # Social media agent can process social media related requests
        return request.data_type in [DataType.TEXT, DataType.SOCIAL_MEDIA, DataType.GENERAL]
    
    @with_error_handling("social_media_processing")
    async def process(self, request: AnalysisRequest) -> AnalysisResult:
        """Process social media requests."""
        try:
            logger.info(f"Processing social media request: {request.data_type}")
            
            start_time = datetime.now()
            
            # Route request based on data type and metadata
            if request.data_type == DataType.SOCIAL_MEDIA:
                result = await self._process_social_media_request(request)
            elif request.data_type == DataType.TEXT:
                result = await self._process_text_request(request)
            else:
                result = await self._process_general_request(request)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return AnalysisResult(
                request_id=request.id,
                data_type=request.data_type,
                status="completed",
                sentiment=result.get("sentiment", SentimentResult(label="neutral", confidence=0.5, reasoning="Social media analysis completed")),
                extracted_text=result.get("extracted_text", ""),
                metadata=result.get("metadata", {}),
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Social media processing failed: {e}")
            return AnalysisResult(
                request_id=request.id,
                data_type=request.data_type,
                status="failed",
                sentiment=SentimentResult(label="neutral", confidence=0.0, reasoning=f"Processing failed: {str(e)}"),
                metadata={"error": str(e)},
                processing_time=0.0
            )
    
    async def _process_social_media_request(self, request: AnalysisRequest) -> Dict[str, Any]:
        """Process social media specific requests."""
        # Extract parameters from request metadata
        platforms = request.metadata.get("platforms", ["twitter", "linkedin"])
        data_types = request.metadata.get("data_types", ["posts", "comments", "sentiment"])
        time_range = request.metadata.get("time_range", "7d")
        include_metadata = request.metadata.get("include_metadata", True)
        
        return await self.platform_manager.integrate_platform_data(
            platforms, data_types, time_range, include_metadata
        )
    
    async def _process_text_request(self, request: AnalysisRequest) -> Dict[str, Any]:
        """Process text-based social media requests."""
        content = request.content
        
        # Analyze social media content
        analysis = {
            "content_type": "social_media_text",
            "sentiment_analysis": await self._analyze_sentiment(content),
            "topic_extraction": await self._extract_topics(content),
            "engagement_prediction": await self._predict_engagement(content)
        }
        
        return analysis
    
    async def _process_general_request(self, request: AnalysisRequest) -> Dict[str, Any]:
        """Process general social media requests."""
        return await self._process_text_request(request)
    
    async def _analyze_sentiment(self, content: str) -> Dict[str, Any]:
        """Analyze sentiment of social media content."""
        # Simple sentiment analysis (would integrate with NLP in production)
        positive_words = ["good", "great", "excellent", "amazing", "love", "like"]
        negative_words = ["bad", "terrible", "awful", "hate", "dislike", "worst"]
        
        content_lower = content.lower()
        positive_count = sum(1 for word in positive_words if word in content_lower)
        negative_count = sum(1 for word in negative_words if word in content_lower)
        
        if positive_count > negative_count:
            sentiment = "positive"
            score = min(0.9, 0.5 + (positive_count - negative_count) * 0.1)
        elif negative_count > positive_count:
            sentiment = "negative"
            score = max(0.1, 0.5 - (negative_count - positive_count) * 0.1)
        else:
            sentiment = "neutral"
            score = 0.5
        
        return {
            "sentiment": sentiment,
            "score": score,
            "positive_words": positive_count,
            "negative_words": negative_count
        }
    
    async def _extract_topics(self, content: str) -> List[Dict[str, Any]]:
        """Extract topics from social media content."""
        # Simple topic extraction (would integrate with NLP in production)
        words = content.lower().split()
        word_freq = {}
        
        for word in words:
            if len(word) > 3 and word.isalpha():
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Return top 5 topics
        sorted_topics = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [{"topic": topic, "frequency": freq} for topic, freq in sorted_topics[:5]]
    
    async def _predict_engagement(self, content: str) -> Dict[str, Any]:
        """Predict engagement for social media content."""
        # Simple engagement prediction (would use ML model in production)
        content_length = len(content)
        has_hashtags = "#" in content
        has_mentions = "@" in content
        has_links = "http" in content
        
        # Simple scoring algorithm
        score = 0.5  # Base score
        
        if 50 <= content_length <= 200:
            score += 0.2  # Optimal length
        elif content_length > 200:
            score -= 0.1  # Too long
        
        if has_hashtags:
            score += 0.1
        
        if has_mentions:
            score += 0.1
        
        if has_links:
            score += 0.05
        
        return {
            "predicted_engagement": min(1.0, max(0.0, score)),
            "factors": {
                "content_length": content_length,
                "has_hashtags": has_hashtags,
                "has_mentions": has_mentions,
                "has_links": has_links
            }
        }
    
    async def integrate_social_media_data(
        self,
        platforms: List[str],
        data_types: List[str],
        time_range: str = "7d",
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """Integrate social media data from multiple platforms."""
        return await self.platform_manager.integrate_platform_data(
            platforms, data_types, time_range, include_metadata
        )
    
    async def analyze_social_media_content(
        self,
        platform: str,
        content_type: str = "posts",
        analysis_type: str = "comprehensive",
        include_engagement: bool = True
    ) -> Dict[str, Any]:
        """Analyze social media content and trends."""
        # Mock implementation - would integrate with actual platform APIs
        return {
            "platform": platform,
            "content_type": content_type,
            "analysis_type": analysis_type,
            "include_engagement": include_engagement,
            "results": {
                "sentiment_distribution": {"positive": 0.6, "neutral": 0.3, "negative": 0.1},
                "top_topics": ["AI", "Technology", "Innovation"],
                "engagement_metrics": {"likes": 150, "shares": 45, "comments": 30},
                "trends": ["#AI", "#Technology", "#Innovation"]
            }
        }
    
    async def monitor_social_media_trends(
        self,
        keywords: List[str],
        platforms: List[str],
        monitoring_period: str = "24h",
        alert_threshold: int = 100
    ) -> Dict[str, Any]:
        """Monitor social media trends and mentions."""
        # Mock implementation - would integrate with actual monitoring APIs
        return {
            "keywords": keywords,
            "platforms": platforms,
            "monitoring_period": monitoring_period,
            "alert_threshold": alert_threshold,
            "results": {
                "total_mentions": 250,
                "trending_keywords": ["AI", "Machine Learning"],
                "platform_breakdown": {
                    "twitter": 120,
                    "linkedin": 80,
                    "facebook": 30,
                    "instagram": 20
                },
                "alerts": [
                    {"keyword": "AI", "mentions": 150, "threshold_exceeded": True},
                    {"keyword": "Machine Learning", "mentions": 100, "threshold_exceeded": False}
                ]
            }
        }
