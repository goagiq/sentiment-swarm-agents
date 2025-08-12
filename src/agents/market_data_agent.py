"""
Market Data Agent for integrating with market research data, news sources, and financial data.
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from src.agents.base_agent import StrandsBaseAgent
from src.core.models import AnalysisRequest, AnalysisResult, DataType, SentimentResult
from src.core.error_handler import with_error_handling


class MarketDataManager:
    """Manage market data integrations and analysis."""
    
    def __init__(self):
        self.data_sources = {}
        self.cache = {}
    
    @with_error_handling("market_data_analysis")
    async def analyze_market_data(
        self,
        market_sector: str,
        data_types: List[str],
        time_range: str = "30d",
        include_competitors: bool = True
    ) -> Dict[str, Any]:
        """Analyze market data and trends."""
        try:
            logger.info(f"Analyzing market data for sector: {market_sector}")
            
            # Collect data from different sources
            market_data = {}
            
            if "sentiment" in data_types:
                market_data["sentiment"] = await self._analyze_market_sentiment(market_sector, time_range)
            
            if "trends" in data_types:
                market_data["trends"] = await self._analyze_market_trends(market_sector, time_range)
            
            if "news" in data_types:
                market_data["news"] = await self._analyze_market_news(market_sector, time_range)
            
            if "social" in data_types:
                market_data["social"] = await self._analyze_social_sentiment(market_sector, time_range)
            
            # Include competitor analysis if requested
            if include_competitors:
                market_data["competitors"] = await self._analyze_competitors(market_sector)
            
            # Combine and analyze all data
            combined_analysis = await self._combine_market_analysis(market_data)
            
            logger.info(f"Market data analysis completed for {market_sector}")
            return {
                "market_sector": market_sector,
                "data_types": data_types,
                "time_range": time_range,
                "include_competitors": include_competitors,
                "market_data": market_data,
                "combined_analysis": combined_analysis
            }
            
        except Exception as e:
            logger.error(f"Market data analysis failed: {e}")
            return {"error": str(e)}
    
    async def _analyze_market_sentiment(self, market_sector: str, time_range: str) -> Dict[str, Any]:
        """Analyze market sentiment."""
        # Mock sentiment analysis
        return {
            "overall_sentiment": 0.75,
            "sentiment_trend": "positive",
            "confidence": 0.85,
            "key_factors": [
                "Strong quarterly earnings",
                "Positive analyst ratings",
                "Market expansion"
            ],
            "sentiment_distribution": {
                "positive": 0.65,
                "neutral": 0.25,
                "negative": 0.10
            }
        }
    
    async def _analyze_market_trends(self, market_sector: str, time_range: str) -> Dict[str, Any]:
        """Analyze market trends."""
        # Mock trend analysis
        return {
            "trend_direction": "upward",
            "trend_strength": 0.8,
            "trend_duration": "3 months",
            "key_trends": [
                "Digital transformation acceleration",
                "Sustainability focus",
                "Remote work adoption"
            ],
            "growth_rate": 0.15,
            "market_size": "1.2T USD"
        }
    
    async def _analyze_market_news(self, market_sector: str, time_range: str) -> Dict[str, Any]:
        """Analyze market news."""
        # Mock news analysis
        return {
            "total_articles": 150,
            "positive_articles": 95,
            "negative_articles": 15,
            "neutral_articles": 40,
            "top_headlines": [
                "Tech Sector Shows Strong Growth",
                "Innovation Drives Market Expansion",
                "New Regulations Impact Industry"
            ],
            "news_sentiment": 0.73
        }
    
    async def _analyze_social_sentiment(self, market_sector: str, time_range: str) -> Dict[str, Any]:
        """Analyze social media sentiment."""
        # Mock social sentiment analysis
        return {
            "social_mentions": 2500,
            "positive_mentions": 1800,
            "negative_mentions": 200,
            "neutral_mentions": 500,
            "social_sentiment": 0.72,
            "trending_topics": [
                "#TechInnovation",
                "#MarketGrowth",
                "#DigitalTransformation"
            ]
        }
    
    async def _analyze_competitors(self, market_sector: str) -> Dict[str, Any]:
        """Analyze competitor landscape."""
        # Mock competitor analysis
        return {
            "top_competitors": [
                {"name": "Competitor A", "market_share": 0.25, "sentiment": 0.68},
                {"name": "Competitor B", "market_share": 0.20, "sentiment": 0.72},
                {"name": "Competitor C", "market_share": 0.15, "sentiment": 0.65}
            ],
            "competitive_landscape": "highly_competitive",
            "market_concentration": 0.60
        }
    
    async def _combine_market_analysis(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Combine all market analysis data."""
        # Calculate overall market health score
        sentiment_score = market_data.get("sentiment", {}).get("overall_sentiment", 0.5)
        trend_score = market_data.get("trends", {}).get("trend_strength", 0.5)
        news_score = market_data.get("news", {}).get("news_sentiment", 0.5)
        social_score = market_data.get("social", {}).get("social_sentiment", 0.5)
        
        overall_score = (sentiment_score + trend_score + news_score + social_score) / 4
        
        return {
            "market_health_score": overall_score,
            "market_outlook": "positive" if overall_score > 0.6 else "neutral" if overall_score > 0.4 else "negative",
            "key_insights": [
                "Strong market sentiment across all channels",
                "Positive trend direction with high confidence",
                "Favorable news coverage and social engagement"
            ],
            "recommendations": [
                "Continue current market strategy",
                "Monitor competitor movements",
                "Focus on innovation and customer engagement"
            ]
        }


class NewsSourceManager:
    """Manage news source integrations and monitoring."""
    
    def __init__(self):
        self.sources = {
            "reuters": ReutersSource(),
            "bloomberg": BloombergSource(),
            "cnn": CNNSource(),
            "bbc": BBCSource()
        }
        self.cache = {}
    
    @with_error_handling("news_monitoring")
    async def monitor_news_sources(
        self,
        sources: List[str],
        keywords: List[str],
        analysis_type: str = "sentiment",
        include_summaries: bool = True
    ) -> Dict[str, Any]:
        """Monitor news sources and headlines."""
        try:
            logger.info(f"Monitoring news sources: {sources}")
            
            all_articles = []
            source_results = {}
            
            # Collect articles from each source
            for source in sources:
                if source in self.sources:
                    articles = await self.sources[source].fetch_articles(keywords)
                    source_results[source] = articles
                    all_articles.extend(articles)
                else:
                    logger.warning(f"News source {source} not supported")
                    source_results[source] = {"error": "Source not supported"}
            
            # Analyze articles based on type
            analysis_results = {}
            
            if analysis_type == "sentiment":
                analysis_results = await self._analyze_news_sentiment(all_articles)
            elif analysis_type == "topics":
                analysis_results = await self._analyze_news_topics(all_articles)
            elif analysis_type == "entities":
                analysis_results = await self._analyze_news_entities(all_articles)
            elif analysis_type == "comprehensive":
                analysis_results = await self._analyze_news_comprehensive(all_articles)
            
            # Generate summaries if requested
            summaries = []
            if include_summaries:
                summaries = await self._generate_news_summaries(all_articles)
            
            logger.info(f"News monitoring completed for {len(sources)} sources")
            return {
                "sources_monitored": sources,
                "keywords": keywords,
                "analysis_type": analysis_type,
                "total_articles": len(all_articles),
                "source_results": source_results,
                "analysis_results": analysis_results,
                "summaries": summaries
            }
            
        except Exception as e:
            logger.error(f"News monitoring failed: {e}")
            return {"error": str(e)}
    
    async def _analyze_news_sentiment(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze sentiment of news articles."""
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        for article in articles:
            sentiment = article.get("sentiment", "neutral")
            if sentiment == "positive":
                positive_count += 1
            elif sentiment == "negative":
                negative_count += 1
            else:
                neutral_count += 1
        
        total = len(articles)
        return {
            "sentiment_distribution": {
                "positive": positive_count / total if total > 0 else 0,
                "negative": negative_count / total if total > 0 else 0,
                "neutral": neutral_count / total if total > 0 else 0
            },
            "overall_sentiment": "positive" if positive_count > negative_count else "negative" if negative_count > positive_count else "neutral"
        }
    
    async def _analyze_news_topics(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze topics in news articles."""
        topics = {}
        for article in articles:
            article_topics = article.get("topics", [])
            for topic in article_topics:
                topics[topic] = topics.get(topic, 0) + 1
        
        # Return top 10 topics
        sorted_topics = sorted(topics.items(), key=lambda x: x[1], reverse=True)
        return {
            "top_topics": [{"topic": topic, "frequency": freq} for topic, freq in sorted_topics[:10]]
        }
    
    async def _analyze_news_entities(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze entities in news articles."""
        entities = {}
        for article in articles:
            article_entities = article.get("entities", [])
            for entity in article_entities:
                entity_name = entity.get("name", "")
                entity_type = entity.get("type", "unknown")
                if entity_name:
                    if entity_name not in entities:
                        entities[entity_name] = {"count": 0, "types": set()}
                    entities[entity_name]["count"] += 1
                    entities[entity_name]["types"].add(entity_type)
        
        # Convert sets to lists for JSON serialization
        for entity_name, data in entities.items():
            data["types"] = list(data["types"])
        
        return {
            "entities": entities
        }
    
    async def _analyze_news_comprehensive(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform comprehensive news analysis."""
        sentiment_analysis = await self._analyze_news_sentiment(articles)
        topic_analysis = await self._analyze_news_topics(articles)
        entity_analysis = await self._analyze_news_entities(articles)
        
        return {
            "sentiment": sentiment_analysis,
            "topics": topic_analysis,
            "entities": entity_analysis,
            "total_articles": len(articles),
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    async def _generate_news_summaries(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate summaries for news articles."""
        summaries = []
        for article in articles[:10]:  # Limit to top 10 articles
            summary = {
                "title": article.get("title", ""),
                "summary": f"Summary of {article.get('title', 'article')}",
                "sentiment": article.get("sentiment", "neutral"),
                "source": article.get("source", "unknown"),
                "published_date": article.get("published_date", "")
            }
            summaries.append(summary)
        
        return summaries


class ReutersSource:
    """Reuters news source integration."""
    
    async def fetch_articles(self, keywords: List[str]) -> List[Dict[str, Any]]:
        """Fetch articles from Reuters."""
        # Mock Reuters articles
        return [
            {
                "title": "Tech Sector Shows Strong Growth",
                "content": "Technology companies report strong quarterly earnings...",
                "sentiment": "positive",
                "topics": ["technology", "earnings", "growth"],
                "entities": [{"name": "Tech Corp", "type": "organization"}],
                "source": "reuters",
                "published_date": "2024-01-15"
            },
            {
                "title": "Market Volatility Continues",
                "content": "Global markets experience continued volatility...",
                "sentiment": "neutral",
                "topics": ["markets", "volatility", "global"],
                "entities": [{"name": "Global Markets", "type": "concept"}],
                "source": "reuters",
                "published_date": "2024-01-14"
            }
        ]


class BloombergSource:
    """Bloomberg news source integration."""
    
    async def fetch_articles(self, keywords: List[str]) -> List[Dict[str, Any]]:
        """Fetch articles from Bloomberg."""
        # Mock Bloomberg articles
        return [
            {
                "title": "Financial Markets Update",
                "content": "Financial markets show positive momentum...",
                "sentiment": "positive",
                "topics": ["finance", "markets", "momentum"],
                "entities": [{"name": "Financial Markets", "type": "concept"}],
                "source": "bloomberg",
                "published_date": "2024-01-15"
            }
        ]


class CNNSource:
    """CNN news source integration."""
    
    async def fetch_articles(self, keywords: List[str]) -> List[Dict[str, Any]]:
        """Fetch articles from CNN."""
        # Mock CNN articles
        return [
            {
                "title": "Business News Roundup",
                "content": "Latest business news and market updates...",
                "sentiment": "neutral",
                "topics": ["business", "news", "markets"],
                "entities": [{"name": "Business News", "type": "concept"}],
                "source": "cnn",
                "published_date": "2024-01-15"
            }
        ]


class BBCSource:
    """BBC news source integration."""
    
    async def fetch_articles(self, keywords: List[str]) -> List[Dict[str, Any]]:
        """Fetch articles from BBC."""
        # Mock BBC articles
        return [
            {
                "title": "Economic Outlook",
                "content": "Economic experts provide outlook for the year...",
                "sentiment": "positive",
                "topics": ["economy", "outlook", "experts"],
                "entities": [{"name": "Economic Experts", "type": "organization"}],
                "source": "bbc",
                "published_date": "2024-01-15"
            }
        ]


class FinancialDataManager:
    """Manage financial data integrations."""
    
    def __init__(self):
        self.data_sources = {}
        self.cache = {}
    
    @with_error_handling("financial_data_integration")
    async def integrate_financial_data(
        self,
        data_source: str,
        symbols: List[str],
        data_types: List[str],
        include_analysis: bool = True
    ) -> Dict[str, Any]:
        """Integrate financial and economic data."""
        try:
            logger.info(f"Integrating financial data from {data_source}")
            
            financial_data = {}
            
            if "price" in data_types:
                financial_data["price"] = await self._fetch_price_data(data_source, symbols)
            
            if "volume" in data_types:
                financial_data["volume"] = await self._fetch_volume_data(data_source, symbols)
            
            if "news" in data_types:
                financial_data["news"] = await self._fetch_financial_news(data_source, symbols)
            
            if "sentiment" in data_types:
                financial_data["sentiment"] = await self._fetch_financial_sentiment(data_source, symbols)
            
            # Perform analysis if requested
            analysis_results = {}
            if include_analysis:
                analysis_results = await self._analyze_financial_data(financial_data)
            
            logger.info(f"Financial data integration completed for {data_source}")
            return {
                "data_source": data_source,
                "symbols": symbols,
                "data_types": data_types,
                "financial_data": financial_data,
                "analysis": analysis_results
            }
            
        except Exception as e:
            logger.error(f"Financial data integration failed: {e}")
            return {"error": str(e)}
    
    async def _fetch_price_data(self, data_source: str, symbols: List[str]) -> Dict[str, Any]:
        """Fetch price data."""
        # Mock price data
        price_data = {}
        for symbol in symbols:
            price_data[symbol] = {
                "current_price": 150.25,
                "change": 2.50,
                "change_percent": 1.69,
                "high": 152.00,
                "low": 148.50,
                "volume": 1000000
            }
        
        return price_data
    
    async def _fetch_volume_data(self, data_source: str, symbols: List[str]) -> Dict[str, Any]:
        """Fetch volume data."""
        # Mock volume data
        volume_data = {}
        for symbol in symbols:
            volume_data[symbol] = {
                "daily_volume": 1000000,
                "avg_volume": 950000,
                "volume_ratio": 1.05
            }
        
        return volume_data
    
    async def _fetch_financial_news(self, data_source: str, symbols: List[str]) -> Dict[str, Any]:
        """Fetch financial news."""
        # Mock financial news
        return {
            "total_articles": 25,
            "positive_articles": 15,
            "negative_articles": 5,
            "neutral_articles": 5,
            "top_headlines": [
                "Strong Earnings Report",
                "Market Rally Continues",
                "Positive Analyst Ratings"
            ]
        }
    
    async def _fetch_financial_sentiment(self, data_source: str, symbols: List[str]) -> Dict[str, Any]:
        """Fetch financial sentiment."""
        # Mock financial sentiment
        return {
            "overall_sentiment": 0.75,
            "sentiment_trend": "positive",
            "confidence": 0.85,
            "sentiment_by_symbol": {
                symbol: {"sentiment": 0.75, "confidence": 0.85} for symbol in symbols
            }
        }
    
    async def _analyze_financial_data(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze financial data."""
        return {
            "market_performance": "positive",
            "volatility_index": 0.15,
            "trend_analysis": "bullish",
            "risk_assessment": "moderate",
            "recommendations": [
                "Monitor market conditions",
                "Consider portfolio diversification",
                "Stay informed of market news"
            ]
        }


class MarketDataAgent(StrandsBaseAgent):
    """
    Market Data Agent for integrating with market research data, news sources, and financial data.
    
    Supports:
    - Market data analysis and trends
    - News source monitoring
    - Financial data integration
    - Economic trend analysis
    """
    
    def __init__(self, model_name: Optional[str] = None, **kwargs):
        super().__init__(model_name=model_name or "mistral-small3.1:latest", **kwargs)
        
        # Initialize market data components
        self.market_data_manager = MarketDataManager()
        self.news_source_manager = NewsSourceManager()
        self.financial_data_manager = FinancialDataManager()
        
        # Set metadata
        self.metadata["agent_type"] = "market_data"
        self.metadata["capabilities"] = [
            "market_analysis",
            "news_monitoring",
            "financial_data",
            "trend_analysis",
            "economic_insights"
        ]
        self.metadata["supported_data_sources"] = ["yahoo_finance", "alpha_vantage", "quandl"]
        self.metadata["supported_news_sources"] = ["reuters", "bloomberg", "cnn", "bbc"]
        
        logger.info("MarketDataAgent initialized successfully")
    
    async def can_process(self, request: AnalysisRequest) -> bool:
        """Check if this agent can process the given request."""
        # Market data agent can process market and financial related requests
        return request.data_type in [DataType.TEXT, DataType.MARKET_DATA, DataType.FINANCIAL, DataType.GENERAL]
    
    @with_error_handling("market_data_processing")
    async def process(self, request: AnalysisRequest) -> AnalysisResult:
        """Process market data requests."""
        try:
            logger.info(f"Processing market data request: {request.data_type}")
            
            start_time = datetime.now()
            
            # Route request based on data type and metadata
            if request.data_type == DataType.MARKET_DATA:
                result = await self._process_market_data_request(request)
            elif request.data_type == DataType.FINANCIAL:
                result = await self._process_financial_request(request)
            elif request.data_type == DataType.TEXT:
                result = await self._process_text_request(request)
            else:
                result = await self._process_general_request(request)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return AnalysisResult(
                request_id=request.id,
                data_type=request.data_type,
                status="completed",
                sentiment=result.get("sentiment", SentimentResult(label="neutral", confidence=0.5, reasoning="Market data analysis completed")),
                extracted_text=result.get("extracted_text", ""),
                metadata=result.get("metadata", {}),
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Market data processing failed: {e}")
            return AnalysisResult(
                request_id=request.id,
                data_type=request.data_type,
                status="failed",
                sentiment=SentimentResult(label="neutral", confidence=0.0, reasoning=f"Processing failed: {str(e)}"),
                metadata={"error": str(e)},
                processing_time=0.0
            )
    
    async def _process_market_data_request(self, request: AnalysisRequest) -> Dict[str, Any]:
        """Process market data related requests."""
        # Extract parameters from request metadata
        market_sector = request.metadata.get("market_sector", "technology")
        data_types = request.metadata.get("data_types", ["sentiment", "trends", "news", "social"])
        time_range = request.metadata.get("time_range", "30d")
        include_competitors = request.metadata.get("include_competitors", True)
        
        return await self.market_data_manager.analyze_market_data(
            market_sector, data_types, time_range, include_competitors
        )
    
    async def _process_financial_request(self, request: AnalysisRequest) -> Dict[str, Any]:
        """Process financial data related requests."""
        # Extract parameters from request metadata
        data_source = request.metadata.get("data_source", "yahoo_finance")
        symbols = request.metadata.get("symbols", ["AAPL", "GOOGL"])
        data_types = request.metadata.get("data_types", ["price", "volume", "news", "sentiment"])
        include_analysis = request.metadata.get("include_analysis", True)
        
        return await self.financial_data_manager.integrate_financial_data(
            data_source, symbols, data_types, include_analysis
        )
    
    async def _process_text_request(self, request: AnalysisRequest) -> Dict[str, Any]:
        """Process text-based market data requests."""
        content = request.content
        
        # Analyze market-related text content
        analysis = {
            "content_type": "market_data_text",
            "market_context": request.metadata.get("market_context", "general"),
            "content_length": len(content),
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        return analysis
    
    async def _process_general_request(self, request: AnalysisRequest) -> Dict[str, Any]:
        """Process general market data requests."""
        return await self._process_text_request(request)
    
    async def analyze_market_data(
        self,
        market_sector: str,
        data_types: List[str],
        time_range: str = "30d",
        include_competitors: bool = True
    ) -> Dict[str, Any]:
        """Analyze market data and trends."""
        return await self.market_data_manager.analyze_market_data(
            market_sector, data_types, time_range, include_competitors
        )
    
    async def monitor_news_sources(
        self,
        sources: List[str],
        keywords: List[str],
        analysis_type: str = "sentiment",
        include_summaries: bool = True
    ) -> Dict[str, Any]:
        """Monitor news sources and headlines."""
        return await self.news_source_manager.monitor_news_sources(
            sources, keywords, analysis_type, include_summaries
        )
    
    async def integrate_financial_data(
        self,
        data_source: str,
        symbols: List[str],
        data_types: List[str],
        include_analysis: bool = True
    ) -> Dict[str, Any]:
        """Integrate financial and economic data."""
        return await self.financial_data_manager.integrate_financial_data(
            data_source, symbols, data_types, include_analysis
        )
