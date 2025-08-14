"""
Social Media Connector for Real-Time Social Data

This module provides real-time social media data integration using various
APIs including Twitter, Reddit, and other social platforms.
"""

import asyncio
import aiohttp
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import json

from loguru import logger

from src.core.error_handler import with_error_handling


@dataclass
class SocialMediaPost:
    """Represents a single social media post."""
    platform: str
    post_id: str
    content: str
    author: str
    timestamp: datetime
    engagement: Dict[str, int]  # likes, shares, comments, etc.
    sentiment_score: float
    confidence: float
    metadata: Dict[str, Any]


class SocialMediaConnector:
    """Connector for real-time social media data from various platforms."""
    
    def __init__(self):
        self.api_keys = {
            "twitter": "demo",  # Replace with actual API key
            "reddit": "demo",   # Replace with actual API key
            "news_api": "demo"  # Replace with actual API key
        }
        
        self.base_urls = {
            "twitter": "https://api.twitter.com/2",
            "reddit": "https://www.reddit.com/api/v1",
            "news_api": "https://newsapi.org/v2"
        }
        
        self.session = None
        self.cache = {}
        self.cache_ttl = 600  # 10 minutes
        
        logger.info("SocialMediaConnector initialized")
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    @with_error_handling("social_media_fetch")
    async def fetch_real_time_data(
        self, 
        keywords: List[str] = None,
        platforms: List[str] = None,
        limit: int = 50
    ) -> List[SocialMediaPost]:
        """
        Fetch real-time social media data for specified keywords.
        
        Args:
            keywords: List of keywords to search for
            platforms: List of platforms to fetch from
            limit: Maximum number of posts to fetch per platform
            
        Returns:
            List of social media posts
        """
        if keywords is None:
            keywords = ["AI", "technology", "business", "finance"]
        
        if platforms is None:
            platforms = ["reddit", "news_api"]
        
        posts = []
        
        for platform in platforms:
            try:
                if platform == "reddit":
                    platform_posts = await self._fetch_reddit_posts(keywords, limit)
                elif platform == "twitter":
                    platform_posts = await self._fetch_twitter_posts(keywords, limit)
                elif platform == "news_api":
                    platform_posts = await self._fetch_news_posts(keywords, limit)
                else:
                    logger.warning(f"Unknown social media platform: {platform}")
                    continue
                
                posts.extend(platform_posts)
                
            except Exception as e:
                logger.error(f"Error fetching data from {platform}: {e}")
                # Fallback to simulated data
                fallback_posts = await self._generate_simulated_posts(
                    keywords, platform, limit
                )
                posts.extend(fallback_posts)
        
        return posts
    
    async def _fetch_reddit_posts(
        self, 
        keywords: List[str], 
        limit: int
    ) -> List[SocialMediaPost]:
        """Fetch posts from Reddit API."""
        posts = []
        
        for keyword in keywords:
            try:
                # Reddit API endpoint for search
                url = f"https://www.reddit.com/search.json"
                params = {
                    "q": keyword,
                    "limit": min(limit, 25),  # Reddit limit
                    "sort": "hot",
                    "t": "day"
                }
                
                headers = {
                    "User-Agent": "SentimentAnalysisBot/1.0"
                }
                
                async with self.session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if "data" in data and "children" in data["data"]:
                            for child in data["data"]["children"]:
                                post_data = child["data"]
                                
                                # Calculate engagement
                                engagement = {
                                    "upvotes": post_data.get("ups", 0),
                                    "downvotes": post_data.get("downs", 0),
                                    "comments": post_data.get("num_comments", 0),
                                    "score": post_data.get("score", 0)
                                }
                                
                                # Simple sentiment calculation based on score
                                sentiment_score = min(max(
                                    (engagement["score"] / 100.0 + 0.5), 0.0
                                ), 1.0)
                                
                                post = SocialMediaPost(
                                    platform="reddit",
                                    post_id=post_data.get("id", ""),
                                    content=post_data.get("title", ""),
                                    author=post_data.get("author", ""),
                                    timestamp=datetime.fromtimestamp(
                                        post_data.get("created_utc", 0)
                                    ),
                                    engagement=engagement,
                                    sentiment_score=sentiment_score,
                                    confidence=0.7,
                                    metadata={
                                        "subreddit": post_data.get("subreddit", ""),
                                        "url": post_data.get("url", ""),
                                        "permalink": post_data.get("permalink", ""),
                                        "keyword": keyword
                                    }
                                )
                                posts.append(post)
                    
            except Exception as e:
                logger.error(f"Error fetching Reddit posts for {keyword}: {e}")
        
        return posts
    
    async def _fetch_twitter_posts(
        self, 
        keywords: List[str], 
        limit: int
    ) -> List[SocialMediaPost]:
        """Fetch posts from Twitter API."""
        posts = []
        
        for keyword in keywords:
            try:
                # Twitter API v2 endpoint for search
                url = f"{self.base_urls['twitter']}/tweets/search/recent"
                params = {
                    "query": keyword,
                    "max_results": min(limit, 100),
                    "tweet.fields": "created_at,public_metrics,author_id"
                }
                
                headers = {
                    "Authorization": f"Bearer {self.api_keys['twitter']}"
                }
                
                async with self.session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if "data" in data:
                            for tweet in data["data"]:
                                metrics = tweet.get("public_metrics", {})
                                
                                engagement = {
                                    "likes": metrics.get("like_count", 0),
                                    "retweets": metrics.get("retweet_count", 0),
                                    "replies": metrics.get("reply_count", 0),
                                    "quotes": metrics.get("quote_count", 0)
                                }
                                
                                # Simple sentiment calculation
                                total_engagement = sum(engagement.values())
                                sentiment_score = min(max(
                                    (total_engagement / 1000.0 + 0.5), 0.0
                                ), 1.0)
                                
                                post = SocialMediaPost(
                                    platform="twitter",
                                    post_id=tweet.get("id", ""),
                                    content=tweet.get("text", ""),
                                    author=tweet.get("author_id", ""),
                                    timestamp=datetime.fromisoformat(
                                        tweet.get("created_at", "").replace("Z", "+00:00")
                                    ),
                                    engagement=engagement,
                                    sentiment_score=sentiment_score,
                                    confidence=0.8,
                                    metadata={
                                        "keyword": keyword,
                                        "lang": tweet.get("lang", "en")
                                    }
                                )
                                posts.append(post)
                    
            except Exception as e:
                logger.error(f"Error fetching Twitter posts for {keyword}: {e}")
        
        return posts
    
    async def _fetch_news_posts(
        self, 
        keywords: List[str], 
        limit: int
    ) -> List[SocialMediaPost]:
        """Fetch news articles from News API."""
        posts = []
        
        for keyword in keywords:
            try:
                url = f"{self.base_urls['news_api']}/everything"
                params = {
                    "q": keyword,
                    "apiKey": self.api_keys["news_api"],
                    "pageSize": min(limit, 100),
                    "sortBy": "publishedAt",
                    "language": "en"
                }
                
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if "articles" in data:
                            for article in data["articles"]:
                                # Calculate engagement (simulated for news)
                                engagement = {
                                    "views": 1000,  # Simulated
                                    "shares": 50,   # Simulated
                                    "comments": 10  # Simulated
                                }
                                
                                # Simple sentiment calculation
                                sentiment_score = 0.6  # Neutral for news
                                
                                post = SocialMediaPost(
                                    platform="news_api",
                                    post_id=article.get("url", ""),
                                    content=article.get("title", ""),
                                    author=article.get("author", ""),
                                    timestamp=datetime.fromisoformat(
                                        article.get("publishedAt", "").replace("Z", "+00:00")
                                    ),
                                    engagement=engagement,
                                    sentiment_score=sentiment_score,
                                    confidence=0.9,
                                    metadata={
                                        "source": article.get("source", {}).get("name", ""),
                                        "url": article.get("url", ""),
                                        "description": article.get("description", ""),
                                        "keyword": keyword
                                    }
                                )
                                posts.append(post)
                    
            except Exception as e:
                logger.error(f"Error fetching news posts for {keyword}: {e}")
        
        return posts
    
    async def _generate_simulated_posts(
        self, 
        keywords: List[str], 
        platform: str, 
        limit: int
    ) -> List[SocialMediaPost]:
        """Generate simulated social media posts as fallback."""
        import random
        
        posts = []
        sample_content = [
            "Great news about {keyword}!",
            "Interesting developments in {keyword}",
            "What do you think about {keyword}?",
            "New insights on {keyword}",
            "Breaking: {keyword} update"
        ]
        
        for keyword in keywords:
            for i in range(min(limit // len(keywords), 5)):
                content = random.choice(sample_content).format(keyword=keyword)
                
                engagement = {
                    "likes": random.randint(10, 1000),
                    "shares": random.randint(5, 500),
                    "comments": random.randint(1, 100)
                }
                
                sentiment_score = random.uniform(0.3, 0.8)
                
                post = SocialMediaPost(
                    platform=f"{platform}_simulated",
                    post_id=f"sim_{platform}_{keyword}_{i}",
                    content=content,
                    author=f"user_{random.randint(1000, 9999)}",
                    timestamp=datetime.now() - timedelta(
                        hours=random.randint(0, 24)
                    ),
                    engagement=engagement,
                    sentiment_score=sentiment_score,
                    confidence=0.3,
                    metadata={
                        "simulated": True,
                        "fallback": True,
                        "keyword": keyword
                    }
                )
                posts.append(post)
        
        return posts
    
    async def get_social_sentiment(
        self, 
        keywords: List[str] = None,
        platforms: List[str] = None
    ) -> Dict[str, float]:
        """Get social sentiment scores for keywords."""
        if keywords is None:
            keywords = ["AI", "technology", "business", "finance"]
        
        if platforms is None:
            platforms = ["reddit", "news_api"]
        
        sentiment_scores = {}
        
        for keyword in keywords:
            try:
                posts = await self.fetch_real_time_data([keyword], platforms)
                
                if posts:
                    # Calculate average sentiment
                    sentiments = [post.sentiment_score for post in posts]
                    avg_sentiment = sum(sentiments) / len(sentiments)
                    sentiment_scores[keyword] = avg_sentiment
                else:
                    sentiment_scores[keyword] = 0.5  # Neutral if no data
                    
            except Exception as e:
                logger.error(f"Error calculating social sentiment for {keyword}: {e}")
                sentiment_scores[keyword] = 0.5
        
        return sentiment_scores
    
    async def get_trending_topics(
        self, 
        platforms: List[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get trending topics from social media platforms."""
        if platforms is None:
            platforms = ["reddit", "news_api"]
        
        trending_topics = []
        
        for platform in platforms:
            try:
                if platform == "reddit":
                    topics = await self._get_reddit_trending(limit)
                elif platform == "news_api":
                    topics = await self._get_news_trending(limit)
                else:
                    continue
                
                trending_topics.extend(topics)
                
            except Exception as e:
                logger.error(f"Error getting trending topics from {platform}: {e}")
        
        return trending_topics
    
    async def _get_reddit_trending(self, limit: int) -> List[Dict[str, Any]]:
        """Get trending topics from Reddit."""
        try:
            url = "https://www.reddit.com/r/all/hot.json"
            params = {"limit": limit}
            headers = {"User-Agent": "SentimentAnalysisBot/1.0"}
            
            async with self.session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    topics = []
                    if "data" in data and "children" in data["data"]:
                        for child in data["data"]["children"]:
                            post_data = child["data"]
                            topics.append({
                                "title": post_data.get("title", ""),
                                "subreddit": post_data.get("subreddit", ""),
                                "score": post_data.get("score", 0),
                                "platform": "reddit"
                            })
                    
                    return topics
        except Exception as e:
            logger.error(f"Error getting Reddit trending topics: {e}")
        
        return []
    
    async def _get_news_trending(self, limit: int) -> List[Dict[str, Any]]:
        """Get trending topics from News API."""
        try:
            url = f"{self.base_urls['news_api']}/top-headlines"
            params = {
                "apiKey": self.api_keys["news_api"],
                "country": "us",
                "pageSize": limit
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    topics = []
                    if "articles" in data:
                        for article in data["articles"]:
                            topics.append({
                                "title": article.get("title", ""),
                                "source": article.get("source", {}).get("name", ""),
                                "published_at": article.get("publishedAt", ""),
                                "platform": "news_api"
                            })
                    
                    return topics
        except Exception as e:
            logger.error(f"Error getting news trending topics: {e}")
        
        return []


# Global social media connector instance
social_media_connector = SocialMediaConnector()


async def get_social_media_connector() -> SocialMediaConnector:
    """Get the global social media connector instance."""
    return social_media_connector
