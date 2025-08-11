"""
Model Management Service for centralized Ollama integration.
Provides unified model management, caching, and fallback mechanisms.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional

import logging

# Configure logger
logger = logging.getLogger(__name__)


class ModelConfig:
    """Configuration for a model."""

    def __init__(
        self,
        model_id: str,
        model_type: str = "text",
        max_tokens: int = 4096,
        temperature: float = 0.7,
        timeout: int = 60,
        retry_attempts: int = 3
    ):
        self.model_id = model_id
        self.model_type = model_type
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.retry_attempts = retry_attempts

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'model_id': self.model_id,
            'model_type': self.model_type,
            'max_tokens': self.max_tokens,
            'temperature': self.temperature,
            'timeout': self.timeout,
            'retry_attempts': self.retry_attempts
        }


class ModelResponse:
    """Response from a model."""

    def __init__(
        self,
        content: str,
        model_id: str,
        processing_time: float,
        tokens_used: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.content = content
        self.model_id = model_id
        self.processing_time = processing_time
        self.tokens_used = tokens_used
        self.metadata = metadata or {}


class ModelManagementService:
    """Service for centralized model management."""

    def __init__(self):
        self.logger = logger
        self.models: Dict[str, ModelConfig] = {}
        self.model_stats = {}
        self._initialize_default_models()

    def _initialize_default_models(self):
        """Initialize default model configurations."""
        default_models = {
            "llama3.2:latest": ModelConfig(
                "llama3.2:latest", "text", 4096, 0.7, 60, 3
            ),
            "llava:latest": ModelConfig(
                "llava:latest", "vision", 4096, 0.7, 120, 3
            ),
            "mistral-small3.1:latest": ModelConfig(
                "mistral-small3.1:latest", "text", 4096, 0.7, 60, 3
            ),
            "llama3.2:3b": ModelConfig(
                "llama3.2:3b", "text", 2048, 0.7, 30, 2
            )
        }

        for model_id, config in default_models.items():
            self.register_model(model_id, config)

    def register_model(self, model_id: str, config: ModelConfig):
        """Register a model configuration."""
        self.models[model_id] = config
        self.model_stats[model_id] = {
            'requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_processing_time': 0.0,
            'average_processing_time': 0.0,
            'last_used': None
        }
        self.logger.info(f"Registered model: {model_id}")

    def get_model_config(self, model_id: str) -> Optional[ModelConfig]:
        """Get model configuration."""
        return self.models.get(model_id)

    def list_models(self) -> List[str]:
        """List all registered models."""
        return list(self.models.keys())

    def get_models_by_type(self, model_type: str) -> List[str]:
        """Get models by type."""
        return [
            model_id for model_id, config in self.models.items()
            if config.model_type == model_type
        ]

    async def generate_text(
        self,
        prompt: str,
        model_id: str,
        **kwargs
    ) -> ModelResponse:
        """Generate text using specified model."""
        config = self.get_model_config(model_id)
        if not config:
            raise ValueError(f"Model not found: {model_id}")

        start_time = time.time()

        try:
            # Update stats
            self.model_stats[model_id]['requests'] += 1
            self.model_stats[model_id]['last_used'] = time.time()

            # Prepare request parameters
            params = {
                'model': model_id,
                'prompt': prompt,
                'stream': False,
                'options': {
                    'num_predict': kwargs.get('max_tokens', config.max_tokens),
                    'temperature': kwargs.get('temperature', config.temperature)
                }
            }

            # Make request to Ollama
            response = await self._make_ollama_request(params, config)

            processing_time = time.time() - start_time

            # Update stats
            self.model_stats[model_id]['successful_requests'] += 1
            self.model_stats[model_id]['total_processing_time'] += processing_time
            self.model_stats[model_id]['average_processing_time'] = (
                self.model_stats[model_id]['total_processing_time'] /
                self.model_stats[model_id]['successful_requests']
            )

            return ModelResponse(
                content=response.get('response', ''),
                model_id=model_id,
                processing_time=processing_time,
                tokens_used=response.get('eval_count'),
                metadata={'raw_response': response}
            )

        except Exception as error:
            processing_time = time.time() - start_time
            self.model_stats[model_id]['failed_requests'] += 1

            self.logger.error(f"Error generating text with {model_id}: {error}")
            raise

    async def generate_vision(
        self,
        prompt: str,
        image_path: str,
        model_id: str,
        **kwargs
    ) -> ModelResponse:
        """Generate text from image using vision model."""
        config = self.get_model_config(model_id)
        if not config:
            raise ValueError(f"Model not found: {model_id}")

        if config.model_type != "vision":
            raise ValueError(f"Model {model_id} is not a vision model")

        start_time = time.time()

        try:
            # Update stats
            self.model_stats[model_id]['requests'] += 1
            self.model_stats[model_id]['last_used'] = time.time()

            # Prepare request parameters
            params = {
                'model': model_id,
                'prompt': prompt,
                'images': [image_path],
                'stream': False,
                'options': {
                    'num_predict': kwargs.get('max_tokens', config.max_tokens),
                    'temperature': kwargs.get('temperature', config.temperature)
                }
            }

            # Make request to Ollama
            response = await self._make_ollama_request(params, config)

            processing_time = time.time() - start_time

            # Update stats
            self.model_stats[model_id]['successful_requests'] += 1
            self.model_stats[model_id]['total_processing_time'] += processing_time
            self.model_stats[model_id]['average_processing_time'] = (
                self.model_stats[model_id]['total_processing_time'] /
                self.model_stats[model_id]['successful_requests']
            )

            return ModelResponse(
                content=response.get('response', ''),
                model_id=model_id,
                processing_time=processing_time,
                tokens_used=response.get('eval_count'),
                metadata={'raw_response': response, 'image_path': image_path}
            )

        except Exception as error:
            processing_time = time.time() - start_time
            self.model_stats[model_id]['failed_requests'] += 1

            self.logger.error(f"Error generating vision with {model_id}: {error}")
            raise

    async def _make_ollama_request(
        self,
        params: Dict[str, Any],
        config: ModelConfig
    ) -> Dict[str, Any]:
        """Make request to Ollama with retry logic."""
        import aiohttp

        for attempt in range(config.retry_attempts):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        'http://localhost:11434/api/generate',
                        json=params,
                        timeout=aiohttp.ClientTimeout(total=config.timeout)
                    ) as response:
                        if response.status == 200:
                            return await response.json()
                        else:
                            error_text = await response.text()
                            raise Exception(f"Ollama error: {error_text}")

            except asyncio.TimeoutError:
                self.logger.warning(
                    f"Timeout on attempt {attempt + 1} for model {config.model_id}"
                )
                if attempt == config.retry_attempts - 1:
                    raise
                await asyncio.sleep(1)

            except Exception as error:
                self.logger.warning(
                    f"Error on attempt {attempt + 1} for model {config.model_id}: {error}"
                )
                if attempt == config.retry_attempts - 1:
                    raise
                await asyncio.sleep(1)

    async def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get information about a model."""
        import aiohttp

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    'http://localhost:11434/api/show',
                    params={'name': model_id},
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        return {'error': f"Failed to get model info: {response.status}"}
        except Exception as error:
            return {'error': f"Error getting model info: {error}"}

    def get_model_stats(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """Get model statistics."""
        if model_id:
            return self.model_stats.get(model_id, {})
        else:
            return self.model_stats

    def get_best_model(self, model_type: str = "text") -> Optional[str]:
        """Get the best performing model of a given type."""
        models_of_type = self.get_models_by_type(model_type)

        if not models_of_type:
            return None

        # Find model with best success rate
        best_model = None
        best_success_rate = 0

        for model_id in models_of_type:
            stats = self.model_stats.get(model_id, {})
            requests = stats.get('requests', 0)
            successful = stats.get('successful_requests', 0)

            if requests > 0:
                success_rate = successful / requests
                if success_rate > best_success_rate:
                    best_success_rate = success_rate
                    best_model = model_id

        return best_model or models_of_type[0]

    async def test_model(self, model_id: str) -> bool:
        """Test if a model is working."""
        try:
            response = await self.generate_text(
                "Hello, this is a test.",
                model_id,
                max_tokens=10
            )
            return bool(response.content)
        except Exception:
            return False

    def reset_stats(self, model_id: Optional[str] = None):
        """Reset model statistics."""
        if model_id:
            if model_id in self.model_stats:
                self.model_stats[model_id] = {
                    'requests': 0,
                    'successful_requests': 0,
                    'failed_requests': 0,
                    'total_processing_time': 0.0,
                    'average_processing_time': 0.0,
                    'last_used': None
                }
        else:
            for model_id in self.model_stats:
                self.reset_stats(model_id)


# Global instance
model_management_service = ModelManagementService()
