"""
Error handling and retry logic with circuit breaker pattern.
Prevents repetitive loops and implements intelligent retry mechanisms.
"""

import time
import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional
from enum import Enum
from dataclasses import dataclass
from functools import wraps
import hashlib

logger = logging.getLogger(__name__)


class ErrorType(Enum):
    """Types of errors that can occur."""
    NETWORK = "network"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    AUTHENTICATION = "authentication"
    VALIDATION = "validation"
    RESOURCE_NOT_FOUND = "resource_not_found"
    SERVER_ERROR = "server_error"
    UNKNOWN = "unknown"


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_backoff: bool = True
    jitter: bool = True
    retry_on_exceptions: tuple = (Exception,)
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 60.0


class CircuitBreaker:
    """Circuit breaker pattern implementation."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
            else:
                raise Exception(f"Circuit breaker is OPEN for {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _on_success(self):
        """Handle successful execution."""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def _on_failure(self):
        """Handle failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"


class RepetitiveLoopDetector:
    """Detects and prevents repetitive loops."""
    
    def __init__(self, max_repetitions: int = 5, time_window: float = 60.0):
        self.max_repetitions = max_repetitions
        self.time_window = time_window
        self.operation_history: Dict[str, List[float]] = {}
    
    def should_continue(self, operation_id: str) -> bool:
        """Check if operation should continue or if it's in a repetitive loop."""
        current_time = time.time()
        
        # Clean old entries
        if operation_id in self.operation_history:
            self.operation_history[operation_id] = [
                t for t in self.operation_history[operation_id]
                if current_time - t < self.time_window
            ]
        
        # Count recent operations
        if operation_id not in self.operation_history:
            self.operation_history[operation_id] = []
        
        recent_count = len(self.operation_history[operation_id])
        
        if recent_count >= self.max_repetitions:
            logger.warning(f"Repetitive loop detected for operation: {operation_id}")
            return False
        
        # Record this operation
        self.operation_history[operation_id].append(current_time)
        return True
    
    def reset_operation(self, operation_id: str):
        """Reset operation history for a specific operation."""
        if operation_id in self.operation_history:
            del self.operation_history[operation_id]


class ErrorHandler:
    """Main error handling class with retry logic and circuit breaker."""
    
    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.loop_detector = RepetitiveLoopDetector()
        self.operation_cache: Dict[str, Any] = {}
    
    def get_circuit_breaker(self, operation_name: str) -> CircuitBreaker:
        """Get or create a circuit breaker for an operation."""
        if operation_name not in self.circuit_breakers:
            self.circuit_breakers[operation_name] = CircuitBreaker(
                failure_threshold=self.config.circuit_breaker_threshold,
                recovery_timeout=self.config.circuit_breaker_timeout
            )
        return self.circuit_breakers[operation_name]
    
    def classify_error(self, error: Exception) -> ErrorType:
        """Classify the type of error."""
        error_str = str(error).lower()
        
        if any(keyword in error_str for keyword in ['timeout', 'timed out']):
            return ErrorType.TIMEOUT
        elif any(keyword in error_str for keyword in ['rate limit', '429', 'too many requests']):
            return ErrorType.RATE_LIMIT
        elif any(keyword in error_str for keyword in ['401', '403', 'unauthorized', 'forbidden']):
            return ErrorType.AUTHENTICATION
        elif any(keyword in error_str for keyword in ['404', 'not found']):
            return ErrorType.RESOURCE_NOT_FOUND
        elif any(keyword in error_str for keyword in ['500', '502', '503', '504', 'server error']):
            return ErrorType.SERVER_ERROR
        elif any(keyword in error_str for keyword in ['network', 'connection', 'dns']):
            return ErrorType.NETWORK
        elif any(keyword in error_str for keyword in ['validation', 'invalid']):
            return ErrorType.VALIDATION
        else:
            return ErrorType.UNKNOWN
    
    def should_retry(self, error: Exception, attempt: int) -> bool:
        """Determine if the operation should be retried."""
        if attempt >= self.config.max_attempts:
            return False
        
        error_type = self.classify_error(error)
        
        # Don't retry certain error types
        if error_type in [ErrorType.AUTHENTICATION, ErrorType.VALIDATION]:
            return False
        
        # Always retry network and timeout errors
        if error_type in [ErrorType.NETWORK, ErrorType.TIMEOUT]:
            return True
        
        # Retry server errors with limits
        if error_type == ErrorType.SERVER_ERROR and attempt < 2:
            return True
        
        # Retry rate limit errors with exponential backoff
        if error_type == ErrorType.RATE_LIMIT:
            return True
        
        return False
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt."""
        if self.config.exponential_backoff:
            delay = self.config.base_delay * (2 ** (attempt - 1))
        else:
            delay = self.config.base_delay
        
        # Add jitter
        if self.config.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)
        
        return min(delay, self.config.max_delay)
    
    def generate_operation_id(self, func: Callable, *args, **kwargs) -> str:
        """Generate a unique operation ID for loop detection."""
        # Create a hash of function name and arguments
        func_name = func.__name__
        args_str = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(f"{func_name}:{args_str}".encode()).hexdigest()
    
    def retry_with_circuit_breaker(self, operation_name: str = None):
        """Decorator for retry logic with circuit breaker."""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                op_name = operation_name or func.__name__
                operation_id = self.generate_operation_id(func, *args, **kwargs)
                
                # Check for repetitive loops
                if not self.loop_detector.should_continue(operation_id):
                    raise Exception(f"Repetitive loop detected for {op_name}. Stopping execution.")
                
                # Check circuit breaker
                circuit_breaker = self.get_circuit_breaker(op_name)
                
                last_error = None
                for attempt in range(1, self.config.max_attempts + 1):
                    try:
                        # Use circuit breaker
                        if asyncio.iscoroutinefunction(func):
                            result = await circuit_breaker.call(func, *args, **kwargs)
                        else:
                            result = circuit_breaker.call(func, *args, **kwargs)
                        
                        # Success - reset operation
                        self.loop_detector.reset_operation(operation_id)
                        return result
                        
                    except Exception as e:
                        last_error = e
                        logger.warning(f"Attempt {attempt} failed for {op_name}: {str(e)}")
                        
                        if not self.should_retry(e, attempt):
                            break
                        
                        if attempt < self.config.max_attempts:
                            delay = self.calculate_delay(attempt)
                            logger.info(f"Retrying {op_name} in {delay:.2f} seconds...")
                            await asyncio.sleep(delay)
                
                # All attempts failed
                self.loop_detector.reset_operation(operation_id)
                raise last_error or Exception(f"All {self.config.max_attempts} attempts failed for {op_name}")
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                op_name = operation_name or func.__name__
                operation_id = self.generate_operation_id(func, *args, **kwargs)
                
                # Check for repetitive loops
                if not self.loop_detector.should_continue(operation_id):
                    raise Exception(f"Repetitive loop detected for {op_name}. Stopping execution.")
                
                # Check circuit breaker
                circuit_breaker = self.get_circuit_breaker(op_name)
                
                last_error = None
                for attempt in range(1, self.config.max_attempts + 1):
                    try:
                        result = circuit_breaker.call(func, *args, **kwargs)
                        
                        # Success - reset operation
                        self.loop_detector.reset_operation(operation_id)
                        return result
                        
                    except Exception as e:
                        last_error = e
                        logger.warning(f"Attempt {attempt} failed for {op_name}: {str(e)}")
                        
                        if not self.should_retry(e, attempt):
                            break
                        
                        if attempt < self.config.max_attempts:
                            delay = self.calculate_delay(attempt)
                            logger.info(f"Retrying {op_name} in {delay:.2f} seconds...")
                            time.sleep(delay)
                
                # All attempts failed
                self.loop_detector.reset_operation(operation_id)
                raise last_error or Exception(f"All {self.config.max_attempts} attempts failed for {op_name}")
            
            # Return appropriate wrapper based on function type
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    def cache_result(self, key: str, result: Any, ttl: float = 300.0):
        """Cache a result with TTL."""
        self.operation_cache[key] = {
            'result': result,
            'expires_at': time.time() + ttl
        }
    
    def get_cached_result(self, key: str) -> Optional[Any]:
        """Get cached result if not expired."""
        if key in self.operation_cache:
            cache_entry = self.operation_cache[key]
            if time.time() < cache_entry['expires_at']:
                return cache_entry['result']
            else:
                del self.operation_cache[key]
        return None
    
    def clear_cache(self):
        """Clear all cached results."""
        self.operation_cache.clear()
    
    def get_status(self) -> Dict[str, Any]:
        """Get error handler status."""
        return {
            'circuit_breakers': {
                name: {
                    'state': cb.state,
                    'failure_count': cb.failure_count,
                    'last_failure_time': cb.last_failure_time
                }
                for name, cb in self.circuit_breakers.items()
            },
            'loop_detector': {
                'active_operations': len(self.loop_detector.operation_history),
                'cached_results': len(self.operation_cache)
            }
        }


# Global error handler instance
error_handler = ErrorHandler()


def with_error_handling(operation_name: str = None, retry_config: Optional[RetryConfig] = None):
    """Decorator for functions that need error handling."""
    if retry_config:
        handler = ErrorHandler(retry_config)
    else:
        handler = error_handler
    
    return handler.retry_with_circuit_breaker(operation_name)


def handle_web_request(url: str, max_retries: int = 3) -> Dict[str, Any]:
    """Specialized handler for web requests with repetitive loop prevention."""
    operation_id = f"web_request:{hashlib.md5(url.encode()).hexdigest()}"
    
    if not error_handler.loop_detector.should_continue(operation_id):
        return {
            "success": False,
            "error": "Repetitive loop detected for web request",
            "suggestion": "Try a different URL or approach"
        }
    
    # Check cache first
    cached_result = error_handler.get_cached_result(operation_id)
    if cached_result:
        return cached_result
    
    # Implement web request logic here
    # This would be integrated with the actual web request functionality
    
    return {
        "success": False,
        "error": "Web request not implemented in this context",
        "suggestion": "Use appropriate web request method"
    }
