"""
API Connector Manager

Provides comprehensive external API integration capabilities including:
- Multiple API endpoint management
- Authentication handling
- Rate limiting and retry logic
- Data transformation and validation
- Error handling and logging
"""

import asyncio
import aiohttp
import requests
import json
import time
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import hashlib
import hmac
from urllib.parse import urlencode
import ssl
import certifi

logger = logging.getLogger(__name__)


class AuthType(Enum):
    """Authentication types supported by the API connector"""
    NONE = "none"
    API_KEY = "api_key"
    BEARER_TOKEN = "bearer_token"
    BASIC_AUTH = "basic_auth"
    OAUTH2 = "oauth2"
    HMAC = "hmac"


@dataclass
class APIConfig:
    """Configuration for an API endpoint"""
    name: str
    base_url: str
    auth_type: AuthType
    auth_credentials: Dict[str, str]
    rate_limit: int = 100  # requests per minute
    timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0
    headers: Dict[str, str] = None
    verify_ssl: bool = True


class APIConnectorManager:
    """
    Manages multiple external API connections with comprehensive features
    """
    
    def __init__(self, configs: Optional[Dict[str, APIConfig]] = None):
        """
        Initialize the API connector manager
        
        Args:
            configs: Dictionary of API configurations
        """
        self.configs = configs or {}
        self.session = None
        self.rate_limit_trackers = {}
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging for the API connector"""
        logging.basicConfig(level=logging.INFO)
        
    def add_api_config(self, name: str, config: APIConfig):
        """
        Add a new API configuration
        
        Args:
            name: Name identifier for the API
            config: API configuration object
        """
        self.configs[name] = config
        self.rate_limit_trackers[name] = {
            'requests': [],
            'last_reset': time.time()
        }
        logger.info(f"Added API configuration for: {name}")
        
    def remove_api_config(self, name: str):
        """
        Remove an API configuration
        
        Args:
            name: Name of the API configuration to remove
        """
        if name in self.configs:
            del self.configs[name]
            if name in self.rate_limit_trackers:
                del self.rate_limit_trackers[name]
            logger.info(f"Removed API configuration for: {name}")
            
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create an aiohttp session"""
        if self.session is None or self.session.closed:
            connector = aiohttp.TCPConnector(
                ssl=ssl.create_default_context(cafile=certifi.where()) if self.configs else None
            )
            self.session = aiohttp.ClientSession(connector=connector)
        return self.session
        
    def _check_rate_limit(self, api_name: str) -> bool:
        """
        Check if we're within rate limits for the API
        
        Args:
            api_name: Name of the API
            
        Returns:
            True if within rate limit, False otherwise
        """
        if api_name not in self.configs:
            return True
            
        config = self.configs[api_name]
        tracker = self.rate_limit_trackers[api_name]
        current_time = time.time()
        
        # Reset tracker if needed
        if current_time - tracker['last_reset'] >= 60:
            tracker['requests'] = []
            tracker['last_reset'] = current_time
            
        # Check if we're at the limit
        if len(tracker['requests']) >= config.rate_limit:
            return False
            
        tracker['requests'].append(current_time)
        return True
        
    def _prepare_headers(self, config: APIConfig, endpoint: str, 
                        method: str, data: Optional[Dict] = None) -> Dict[str, str]:
        """
        Prepare headers for the API request
        
        Args:
            config: API configuration
            endpoint: API endpoint
            method: HTTP method
            data: Request data for HMAC signing
            
        Returns:
            Dictionary of headers
        """
        headers = config.headers.copy() if config.headers else {}
        
        if config.auth_type == AuthType.API_KEY:
            headers['X-API-Key'] = config.auth_credentials.get('api_key', '')
        elif config.auth_type == AuthType.BEARER_TOKEN:
            headers['Authorization'] = f"Bearer {config.auth_credentials.get('token', '')}"
        elif config.auth_type == AuthType.BASIC_AUTH:
            import base64
            username = config.auth_credentials.get('username', '')
            password = config.auth_credentials.get('password', '')
            credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
            headers['Authorization'] = f"Basic {credentials}"
        elif config.auth_type == AuthType.HMAC:
            # Generate HMAC signature
            secret = config.auth_credentials.get('secret', '')
            timestamp = str(int(time.time()))
            message = f"{method}{endpoint}{timestamp}"
            if data:
                message += json.dumps(data, sort_keys=True)
            
            signature = hmac.new(
                secret.encode(),
                message.encode(),
                hashlib.sha256
            ).hexdigest()
            
            headers['X-Timestamp'] = timestamp
            headers['X-Signature'] = signature
            
        return headers
        
    async def make_request(self, api_name: str, endpoint: str, 
                          method: str = "GET", data: Optional[Dict] = None,
                          params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Make an API request with comprehensive error handling
        
        Args:
            api_name: Name of the API configuration to use
            endpoint: API endpoint
            method: HTTP method
            data: Request data
            params: Query parameters
            
        Returns:
            Response data dictionary
        """
        if api_name not in self.configs:
            raise ValueError(f"API configuration '{api_name}' not found")
            
        config = self.configs[api_name]
        
        # Check rate limit
        if not self._check_rate_limit(api_name):
            raise Exception(f"Rate limit exceeded for API: {api_name}")
            
        # Prepare request
        url = f"{config.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        headers = self._prepare_headers(config, endpoint, method, data)
        
        session = await self._get_session()
        
        for attempt in range(config.retry_attempts):
            try:
                async with session.request(
                    method=method,
                    url=url,
                    headers=headers,
                    json=data if data else None,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=config.timeout),
                    ssl=config.verify_ssl
                ) as response:
                    
                    if response.status >= 400:
                        error_text = await response.text()
                        logger.error(f"API request failed: {response.status} - {error_text}")
                        raise Exception(f"API request failed: {response.status} - {error_text}")
                        
                    content_type = response.headers.get('content-type', '')
                    if 'application/json' in content_type:
                        result = await response.json()
                    else:
                        result = await response.text()
                        
                    logger.info(f"API request successful: {api_name} - {endpoint}")
                    return {
                        'status': response.status,
                        'data': result,
                        'headers': dict(response.headers)
                    }
                    
            except asyncio.TimeoutError:
                logger.warning(f"API request timeout (attempt {attempt + 1}): {api_name}")
                if attempt == config.retry_attempts - 1:
                    raise Exception(f"API request timeout after {config.retry_attempts} attempts")
                await asyncio.sleep(config.retry_delay * (2 ** attempt))
                
            except Exception as e:
                logger.error(f"API request error (attempt {attempt + 1}): {str(e)}")
                if attempt == config.retry_attempts - 1:
                    raise
                await asyncio.sleep(config.retry_delay * (2 ** attempt))
                
    def make_sync_request(self, api_name: str, endpoint: str,
                         method: str = "GET", data: Optional[Dict] = None,
                         params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Make a synchronous API request
        
        Args:
            api_name: Name of the API configuration to use
            endpoint: API endpoint
            method: HTTP method
            data: Request data
            params: Query parameters
            
        Returns:
            Response data dictionary
        """
        if api_name not in self.configs:
            raise ValueError(f"API configuration '{api_name}' not found")
            
        config = self.configs[api_name]
        
        # Check rate limit
        if not self._check_rate_limit(api_name):
            raise Exception(f"Rate limit exceeded for API: {api_name}")
            
        # Prepare request
        url = f"{config.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        headers = self._prepare_headers(config, endpoint, method, data)
        
        for attempt in range(config.retry_attempts):
            try:
                response = requests.request(
                    method=method,
                    url=url,
                    headers=headers,
                    json=data if data else None,
                    params=params,
                    timeout=config.timeout,
                    verify=config.verify_ssl
                )
                
                if response.status_code >= 400:
                    logger.error(f"API request failed: {response.status_code} - {response.text}")
                    raise Exception(f"API request failed: {response.status_code} - {response.text}")
                    
                content_type = response.headers.get('content-type', '')
                if 'application/json' in content_type:
                    result = response.json()
                else:
                    result = response.text
                    
                logger.info(f"API request successful: {api_name} - {endpoint}")
                return {
                    'status': response.status_code,
                    'data': result,
                    'headers': dict(response.headers)
                }
                
            except requests.exceptions.Timeout:
                logger.warning(f"API request timeout (attempt {attempt + 1}): {api_name}")
                if attempt == config.retry_attempts - 1:
                    raise Exception(f"API request timeout after {config.retry_attempts} attempts")
                time.sleep(config.retry_delay * (2 ** attempt))
                
            except Exception as e:
                logger.error(f"API request error (attempt {attempt + 1}): {str(e)}")
                if attempt == config.retry_attempts - 1:
                    raise
                time.sleep(config.retry_delay * (2 ** attempt))
                
    async def batch_request(self, api_name: str, requests_list: List[Dict]) -> List[Dict]:
        """
        Make multiple API requests in batch
        
        Args:
            api_name: Name of the API configuration to use
            requests_list: List of request dictionaries
            
        Returns:
            List of response dictionaries
        """
        tasks = []
        for req in requests_list:
            task = self.make_request(
                api_name=api_name,
                endpoint=req.get('endpoint', ''),
                method=req.get('method', 'GET'),
                data=req.get('data'),
                params=req.get('params')
            )
            tasks.append(task)
            
        return await asyncio.gather(*tasks, return_exceptions=True)
        
    def get_api_status(self, api_name: str) -> Dict[str, Any]:
        """
        Get status information for an API
        
        Args:
            api_name: Name of the API
            
        Returns:
            Status information dictionary
        """
        if api_name not in self.configs:
            return {'error': f"API '{api_name}' not found"}
            
        config = self.configs[api_name]
        tracker = self.rate_limit_trackers[api_name]
        
        return {
            'name': api_name,
            'base_url': config.base_url,
            'auth_type': config.auth_type.value,
            'rate_limit': config.rate_limit,
            'requests_this_minute': len(tracker['requests']),
            'timeout': config.timeout,
            'retry_attempts': config.retry_attempts
        }
        
    async def close(self):
        """Close the aiohttp session"""
        if self.session and not self.session.closed:
            await self.session.close()
            
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.session and not self.session.closed:
            asyncio.create_task(self.close())
