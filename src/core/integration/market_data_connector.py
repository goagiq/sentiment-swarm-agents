"""
Market Data Connector for Real-Time Financial Data

This module provides real-time market data integration using various
financial APIs including Alpha Vantage, Yahoo Finance, and others.
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
class MarketDataPoint:
    """Represents a single market data point."""
    symbol: str
    price: float
    volume: int
    change: float
    change_percent: float
    timestamp: datetime
    source: str
    confidence: float
    metadata: Dict[str, Any]


class MarketDataConnector:
    """Connector for real-time market data from various sources."""
    
    def __init__(self):
        self.api_keys = {
            "alpha_vantage": "demo",  # Replace with actual API key
            "yahoo_finance": None,  # No API key needed
            "finnhub": "demo"  # Replace with actual API key
        }
        
        self.base_urls = {
            "alpha_vantage": "https://www.alphavantage.co/query",
            "yahoo_finance": "https://query1.finance.yahoo.com/v8/finance",
            "finnhub": "https://finnhub.io/api/v1"
        }
        
        self.session = None
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
        
        logger.info("MarketDataConnector initialized")
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    @with_error_handling("market_data_fetch")
    async def fetch_real_time_data(
        self, 
        symbols: List[str] = None,
        sources: List[str] = None
    ) -> List[MarketDataPoint]:
        """
        Fetch real-time market data for specified symbols.
        
        Args:
            symbols: List of stock symbols to fetch
            sources: List of data sources to use
            
        Returns:
            List of market data points
        """
        if symbols is None:
            symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
        
        if sources is None:
            sources = ["yahoo_finance", "alpha_vantage"]
        
        data_points = []
        
        for source in sources:
            try:
                if source == "yahoo_finance":
                    source_data = await self._fetch_yahoo_finance(symbols)
                elif source == "alpha_vantage":
                    source_data = await self._fetch_alpha_vantage(symbols)
                elif source == "finnhub":
                    source_data = await self._fetch_finnhub(symbols)
                else:
                    logger.warning(f"Unknown market data source: {source}")
                    continue
                
                data_points.extend(source_data)
                
            except Exception as e:
                logger.error(f"Error fetching data from {source}: {e}")
                # Fallback to simulated data
                fallback_data = await self._generate_simulated_data(symbols, source)
                data_points.extend(fallback_data)
        
        return data_points
    
    async def _fetch_yahoo_finance(self, symbols: List[str]) -> List[MarketDataPoint]:
        """Fetch data from Yahoo Finance API."""
        data_points = []
        
        for symbol in symbols:
            try:
                url = f"{self.base_urls['yahoo_finance']}/chart/{symbol}"
                params = {
                    "interval": "1m",
                    "range": "1d",
                    "includePrePost": "false"
                }
                
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if "chart" in data and "result" in data["chart"]:
                            result = data["chart"]["result"][0]
                            meta = result.get("meta", {})
                            timestamp = result.get("timestamp", [])
                            indicators = result.get("indicators", {})
                            
                            if timestamp and indicators.get("quote"):
                                quote = indicators["quote"][0]
                                prices = quote.get("close", [])
                                volumes = quote.get("volume", [])
                                
                                if prices and volumes:
                                    current_price = prices[-1]
                                    current_volume = volumes[-1]
                                    previous_price = prices[-2] if len(prices) > 1 else current_price
                                    
                                    change = current_price - previous_price
                                    change_percent = (change / previous_price * 100) if previous_price > 0 else 0
                                    
                                    data_point = MarketDataPoint(
                                        symbol=symbol,
                                        price=current_price,
                                        volume=current_volume,
                                        change=change,
                                        change_percent=change_percent,
                                        timestamp=datetime.fromtimestamp(timestamp[-1]),
                                        source="yahoo_finance",
                                        confidence=0.9,
                                        metadata={
                                            "currency": meta.get("currency", "USD"),
                                            "exchange": meta.get("exchangeName", "NMS"),
                                            "market_state": meta.get("marketState", "REGULAR")
                                        }
                                    )
                                    data_points.append(data_point)
                    
            except Exception as e:
                logger.error(f"Error fetching Yahoo Finance data for {symbol}: {e}")
        
        return data_points
    
    async def _fetch_alpha_vantage(self, symbols: List[str]) -> List[MarketDataPoint]:
        """Fetch data from Alpha Vantage API."""
        data_points = []
        
        for symbol in symbols:
            try:
                url = self.base_urls["alpha_vantage"]
                params = {
                    "function": "GLOBAL_QUOTE",
                    "symbol": symbol,
                    "apikey": self.api_keys["alpha_vantage"]
                }
                
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if "Global Quote" in data:
                            quote = data["Global Quote"]
                            
                            data_point = MarketDataPoint(
                                symbol=symbol,
                                price=float(quote.get("05. price", 0)),
                                volume=int(quote.get("06. volume", 0)),
                                change=float(quote.get("09. change", 0)),
                                change_percent=float(quote.get("10. change percent", "0%").replace("%", "")),
                                timestamp=datetime.now(),
                                source="alpha_vantage",
                                confidence=0.85,
                                metadata={
                                    "currency": quote.get("08. previous close", "USD"),
                                    "exchange": "NYSE",
                                    "market_state": "REGULAR"
                                }
                            )
                            data_points.append(data_point)
                    
            except Exception as e:
                logger.error(f"Error fetching Alpha Vantage data for {symbol}: {e}")
        
        return data_points
    
    async def _fetch_finnhub(self, symbols: List[str]) -> List[MarketDataPoint]:
        """Fetch data from Finnhub API."""
        data_points = []
        
        for symbol in symbols:
            try:
                url = f"{self.base_urls['finnhub']}/quote"
                params = {
                    "symbol": symbol,
                    "token": self.api_keys["finnhub"]
                }
                
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if "c" in data:  # Current price
                            data_point = MarketDataPoint(
                                symbol=symbol,
                                price=data["c"],
                                volume=data.get("v", 0),
                                change=data.get("d", 0),
                                change_percent=data.get("dp", 0),
                                timestamp=datetime.fromtimestamp(data.get("t", datetime.now().timestamp())),
                                source="finnhub",
                                confidence=0.8,
                                metadata={
                                    "high": data.get("h"),
                                    "low": data.get("l"),
                                    "open": data.get("o"),
                                    "previous_close": data.get("pc")
                                }
                            )
                            data_points.append(data_point)
                    
            except Exception as e:
                logger.error(f"Error fetching Finnhub data for {symbol}: {e}")
        
        return data_points
    
    async def _generate_simulated_data(
        self, 
        symbols: List[str], 
        source: str
    ) -> List[MarketDataPoint]:
        """Generate simulated market data as fallback."""
        import random
        
        data_points = []
        base_prices = {
            "AAPL": 150.0, "GOOGL": 2800.0, "MSFT": 300.0,
            "TSLA": 800.0, "AMZN": 3300.0
        }
        
        for symbol in symbols:
            base_price = base_prices.get(symbol, 100.0)
            price_change = random.uniform(-5.0, 5.0)
            current_price = base_price + price_change
            change_percent = (price_change / base_price) * 100
            
            data_point = MarketDataPoint(
                symbol=symbol,
                price=round(current_price, 2),
                volume=random.randint(1000000, 10000000),
                change=round(price_change, 2),
                change_percent=round(change_percent, 2),
                timestamp=datetime.now(),
                source=f"{source}_simulated",
                confidence=0.5,
                metadata={
                    "simulated": True,
                    "fallback": True
                }
            )
            data_points.append(data_point)
        
        return data_points
    
    async def get_market_sentiment(self, symbols: List[str] = None) -> Dict[str, float]:
        """Get market sentiment scores for symbols."""
        if symbols is None:
            symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
        
        sentiment_scores = {}
        
        for symbol in symbols:
            try:
                # Calculate sentiment based on price movement
                data_points = await self.fetch_real_time_data([symbol])
                
                if data_points:
                    latest = data_points[0]
                    
                    # Simple sentiment calculation based on price change
                    if latest.change_percent > 2:
                        sentiment = 0.8  # Very positive
                    elif latest.change_percent > 0:
                        sentiment = 0.6  # Positive
                    elif latest.change_percent > -2:
                        sentiment = 0.4  # Neutral
                    else:
                        sentiment = 0.2  # Negative
                    
                    sentiment_scores[symbol] = sentiment
                else:
                    sentiment_scores[symbol] = 0.5  # Neutral if no data
                    
            except Exception as e:
                logger.error(f"Error calculating sentiment for {symbol}: {e}")
                sentiment_scores[symbol] = 0.5
        
        return sentiment_scores
    
    async def get_market_volatility(self, symbols: List[str] = None) -> Dict[str, float]:
        """Get market volatility scores for symbols."""
        if symbols is None:
            symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
        
        volatility_scores = {}
        
        for symbol in symbols:
            try:
                # Calculate volatility based on price change percentage
                data_points = await self.fetch_real_time_data([symbol])
                
                if data_points:
                    latest = data_points[0]
                    
                    # Simple volatility calculation
                    volatility = min(abs(latest.change_percent) / 10.0, 1.0)
                    volatility_scores[symbol] = volatility
                else:
                    volatility_scores[symbol] = 0.1  # Low volatility if no data
                    
            except Exception as e:
                logger.error(f"Error calculating volatility for {symbol}: {e}")
                volatility_scores[symbol] = 0.1
        
        return volatility_scores
    



# Global market data connector instance
market_data_connector = MarketDataConnector()


async def get_market_data_connector() -> MarketDataConnector:
    """Get the global market data connector instance."""
    return market_data_connector
