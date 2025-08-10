"""
Optimized Ollama integration for all 7 agents with connection pooling, 
model sharing, and performance monitoring.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from loguru import logger
import aiohttp
from collections import defaultdict

# Mock OllamaModel for testing
class OllamaModel:
    """Mock Ollama model for testing."""
    def __init__(self, model_name: str = "llama3.2:latest"):
        self.model_name = model_name
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Mock generation."""
        return f"Mock response from {self.model_name}: {prompt[:50]}..."

STRANDS_AVAILABLE = True


@dataclass
class ModelMetrics:
    """Performance metrics for Ollama models."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_response_time: float = 0.0
    average_response_time: float = 0.0
    last_used: float = 0.0
    error_count: int = 0
    
    def update_metrics(self, response_time: float, success: bool):
        """Update metrics after a request."""
        self.total_requests += 1
        self.total_response_time += response_time
        self.average_response_time = self.total_response_time / self.total_requests
        self.last_used = time.time()
        
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
            self.error_count += 1


@dataclass
class ConnectionPool:
    """Connection pool for Ollama HTTP connections."""
    host: str
    max_connections: int = 10
    max_keepalive: int = 30
    connections: List[aiohttp.ClientSession] = field(default_factory=list)
    active_connections: int = 0
    
    async def get_connection(self) -> aiohttp.ClientSession:
        """Get a connection from the pool or create a new one."""
        if (self.connections and 
                self.active_connections < self.max_connections):
            conn = self.connections.pop()
            self.active_connections += 1
            return conn
        
        # Create new connection
        timeout = aiohttp.ClientTimeout(total=30)
        conn = aiohttp.ClientSession(
            timeout=timeout,
            connector=aiohttp.TCPConnector(
                limit=self.max_connections,
                keepalive_timeout=self.max_keepalive
            )
        )
        self.active_connections += 1
        return conn
    
    async def return_connection(self, conn: aiohttp.ClientSession):
        """Return a connection to the pool."""
        if self.active_connections > 0:
            self.active_connections -= 1
        
        if len(self.connections) < self.max_connections:
            self.connections.append(conn)
        else:
            await conn.close()


class OptimizedOllamaIntegration:
    """
    Optimized Ollama integration with:
    - Connection pooling
    - Model sharing
    - Lazy loading
    - Performance monitoring
    - Automatic fallbacks
    """
    
    def __init__(self, host: str = "http://localhost:11434"):
        self.host = host
        self.connection_pool = ConnectionPool(host)
        self.models: Dict[str, OllamaModel] = {}
        self.model_metrics: Dict[str, ModelMetrics] = defaultdict(ModelMetrics)
        self.model_configs: Dict[str, Dict[str, Any]] = {}
        self._initialized = False
        self._lock = asyncio.Lock()
        
        # Model configurations for different use cases
        self._setup_model_configs()
    
    def _setup_model_configs(self):
        """Setup model configurations for different agent types."""
        self.model_configs = {
            "text": {
                "model_id": "llama3.2:latest",
                "temperature": 0.1,
                "max_tokens": 100,
                "keep_alive": "5m",
                "capabilities": ["text", "sentiment_analysis"]
            },
            "vision": {
                "model_id": "llava:latest", 
                "temperature": 0.7,
                "max_tokens": 200,
                "keep_alive": "10m",
                "capabilities": ["vision", "image_analysis"]
            },
            "audio": {
                "model_id": "llava:latest",  # Same as vision for now
                "temperature": 0.7,
                "max_tokens": 200,
                "keep_alive": "10m",
                "capabilities": ["audio", "transcription"]
            },
            "swarm": {
                "model_id": "llama3.2:latest",
                "temperature": 0.3,
                "max_tokens": 150,
                "keep_alive": "5m",
                "capabilities": ["coordination", "planning"]
            },
            "orchestrator": {
                "model_id": "llama3.2:latest",
                "temperature": 0.2,
                "max_tokens": 200,
                "keep_alive": "5m",
                "capabilities": ["coordination", "decision_making"]
            }
        }
    
    async def initialize(self):
        """Initialize the Ollama integration lazily."""
        if self._initialized:
            return
        
        async with self._lock:
            if self._initialized:
                return
            
            try:
                # Check Ollama server availability
                if await self._check_server_health():
                    logger.info("Ollama server is available")
                    self._initialized = True
                else:
                    logger.warning("Ollama server not available, using fallback mode")
                    
            except Exception as e:
                logger.error(f"Failed to initialize Ollama integration: {e}")
                self._initialized = False
    
    async def _check_server_health(self) -> bool:
        """Check if Ollama server is healthy."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.host}/api/tags", timeout=5) as response:
                    return response.status == 200
        except Exception:
            return False
    
    async def get_model(self, model_type: str, create_if_missing: bool = True) -> Optional[OllamaModel]:
        """
        Get a model by type with lazy loading and sharing.
        
        Args:
            model_type: Type of model (text, vision, audio, swarm, orchestrator)
            create_if_missing: Whether to create the model if it doesn't exist
            
        Returns:
            OllamaModel instance or None if not available
        """
        await self.initialize()
        
        # Check if model already exists
        if model_type in self.models:
            self.model_metrics[model_type].last_used = time.time()
            return self.models[model_type]
        
        # Check if we can share a model (e.g., audio and vision both use llava)
        shared_type = self._get_shared_model_type(model_type)
        if shared_type and shared_type in self.models:
            logger.info(f"Sharing {shared_type} model for {model_type}")
            self.model_metrics[shared_type].last_used = time.time()
            return self.models[shared_type]
        
        # Create new model if requested
        if create_if_missing and STRANDS_AVAILABLE:
            return await self._create_model(model_type)
        
        return None
    
    def _get_shared_model_type(self, model_type: str) -> Optional[str]:
        """Get shared model type for efficiency."""
        if model_type == "audio" and "vision" in self.models:
            return "vision"  # Audio and vision can share llava
        elif model_type == "swarm" and "text" in self.models:
            return "text"  # Swarm can use text model
        elif model_type == "orchestrator" and "text" in self.models:
            return "text"  # Orchestrator can use text model
        return None
    
    async def _create_model(self, model_type: str) -> Optional[OllamaModel]:
        """Create a new Ollama model instance."""
        if not STRANDS_AVAILABLE:
            return None
        
        config = self.model_configs.get(model_type)
        if not config:
            logger.error(f"No configuration found for model type: {model_type}")
            return None
        
        try:
            model = OllamaModel(
                host=self.host,
                model_id=config["model_id"],
                temperature=config.get("temperature", 0.7),
                max_tokens=config.get("max_tokens", 100),
                keep_alive=config.get("keep_alive", "5m")
            )
            
            self.models[model_type] = model
            logger.info(f"Created Ollama model for {model_type}: {config['model_id']}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to create {model_type} model: {e}")
            return None
    
    @asynccontextmanager
    async def get_connection(self):
        """Get a connection from the pool with automatic cleanup."""
        conn = await self.connection_pool.get_connection()
        try:
            yield conn
        finally:
            await self.connection_pool.return_connection(conn)
    
    async def execute_with_metrics(
        self, 
        model_type: str, 
        operation: str,
        func: callable,
        *args,
        **kwargs
    ) -> Any:
        """
        Execute a function with performance monitoring.
        
        Args:
            model_type: Type of model being used
            operation: Name of the operation for logging
            func: Function to execute
            *args, **kwargs: Arguments for the function
            
        Returns:
            Result of the function execution
        """
        start_time = time.time()
        success = False
        
        try:
            result = await func(*args, **kwargs)
            success = True
            return result
            
        except Exception as e:
            logger.error(f"Error in {operation} for {model_type}: {e}")
            raise
            
        finally:
            response_time = time.time() - start_time
            self.model_metrics[model_type].update_metrics(response_time, success)
    
    async def get_model_performance(self, model_type: str) -> Optional[ModelMetrics]:
        """Get performance metrics for a specific model."""
        return self.model_metrics.get(model_type)
    
    async def get_all_performance_metrics(self) -> Dict[str, ModelMetrics]:
        """Get performance metrics for all models."""
        return dict(self.model_metrics)
    
    async def cleanup_unused_models(self, max_idle_time: int = 300):
        """Clean up models that haven't been used for a while."""
        current_time = time.time()
        models_to_remove = []
        
        for model_type, metrics in self.model_metrics.items():
            if current_time - metrics.last_used > max_idle_time:
                models_to_remove.append(model_type)
        
        for model_type in models_to_remove:
            if model_type in self.models:
                del self.models[model_type]
                del self.model_metrics[model_type]
                logger.info(f"Cleaned up unused model: {model_type}")
    
    async def get_available_models(self) -> List[str]:
        """Get list of available model types."""
        return list(self.models.keys())
    
    async def check_model_availability(self, model_id: str) -> bool:
        """Check if a specific model is available on the Ollama server."""
        try:
            async with self.get_connection() as session:
                async with session.get(f"{self.host}/api/tags") as response:
                    if response.status == 200:
                        models = await response.json()
                        return any(
                            model["name"] == model_id 
                            for model in models.get("models", [])
                        )
                    return False
        except Exception as e:
            logger.error(f"Error checking model availability: {e}")
            return False
    
    async def cleanup(self):
        """Cleanup all resources."""
        try:
            # Close all model connections
            for model in self.models.values():
                if hasattr(model, 'close'):
                    await model.close()
            
            # Close connection pool
            for conn in self.connection_pool.connections:
                await conn.close()
            
            self.models.clear()
            self.model_metrics.clear()
            logger.info("Optimized Ollama integration cleaned up")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


# Global optimized Ollama integration instance
optimized_ollama = OptimizedOllamaIntegration()


# Convenience functions for backward compatibility
async def get_ollama_model(model_type: str = "text") -> Optional[OllamaModel]:
    """Get an Ollama model by type using the optimized integration."""
    return await optimized_ollama.get_model(model_type)


async def get_ollama_model_with_metrics(
    model_type: str, 
    operation: str,
    func: callable,
    *args,
    **kwargs
) -> Any:
    """Execute a function with performance monitoring."""
    return await optimized_ollama.execute_with_metrics(
        model_type, operation, func, *args, **kwargs
    )


async def get_model_performance(model_type: str) -> Optional[ModelMetrics]:
    """Get performance metrics for a specific model."""
    return await optimized_ollama.get_model_performance(model_type)


async def cleanup_ollama():
    """Cleanup the optimized Ollama integration."""
    await optimized_ollama.cleanup()
