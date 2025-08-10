# Error Handling and Repetitive Loop Prevention Guide

## Overview

This guide explains how the error handling system prevents repetitive loops and implements intelligent retry mechanisms in the Sentiment Analysis Swarm application.

## Problem Solved

The original implementation had a critical issue where the system would get stuck in repetitive loops when trying to fetch web content. For example:

```python
# PROBLEMATIC PATTERN (Before Fix)
for attempt in range(10):  # Could run indefinitely
    try:
        result = fetch_web_content(url)
        return result
    except Exception as e:
        print(f"Attempt {attempt} failed: {e}")
        # No intelligent handling - just keeps retrying
```

This pattern would:
- Continue retrying the same failed operation indefinitely
- Waste system resources
- Provide poor user experience
- Not learn from failures

## Solution Implemented

### 1. Circuit Breaker Pattern

The system implements a circuit breaker pattern that prevents cascading failures:

```python
class CircuitBreaker:
    def __init__(self, failure_threshold=5, recovery_timeout=60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
```

**States:**
- **CLOSED**: Normal operation, requests pass through
- **OPEN**: Circuit is open, requests fail fast
- **HALF_OPEN**: Testing if service has recovered

### 2. Repetitive Loop Detection

The `RepetitiveLoopDetector` tracks operation attempts and prevents infinite loops:

```python
class RepetitiveLoopDetector:
    def __init__(self, max_repetitions=5, time_window=60.0):
        self.max_repetitions = max_repetitions
        self.time_window = time_window
        self.operation_history = {}
    
    def should_continue(self, operation_id: str) -> bool:
        # Check if operation has been attempted too many times
        # Return False if repetitive loop detected
```

### 3. Intelligent Error Classification

The system classifies errors to determine appropriate retry strategies:

```python
class ErrorType(Enum):
    NETWORK = "network"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    AUTHENTICATION = "authentication"
    VALIDATION = "validation"
    RESOURCE_NOT_FOUND = "resource_not_found"
    SERVER_ERROR = "server_error"
    UNKNOWN = "unknown"
```

## Integration with Main Program

### 1. Import Error Handling

```python
# In main.py or any module
from core.error_handler import error_handler, with_error_handling, RetryConfig
```

### 2. Apply to MCP Tools

```python
@self.mcp.tool(description="Analyze text sentiment using TextAgent")
@with_error_handling("text_sentiment_analysis")
async def analyze_text_sentiment(text: str, language: str = "en"):
    """Analyze text sentiment using TextAgent."""
    analysis_request = AnalysisRequest(
        data_type=DataType.TEXT,
        content=text,
        language=language
    )
    
    result = await self.agents["text"].process(analysis_request)
    
    return {
        "success": True,
        "agent": "text",
        "sentiment": result.sentiment.label,
        "confidence": result.sentiment.confidence,
        "processing_time": result.processing_time
    }
```

### 3. Safe Web Request Pattern

```python
def safe_web_request(url: str, max_retries: int = 3) -> Dict[str, Any]:
    """Safe web request with repetitive loop prevention."""
    operation_name = f"web_request_{hash(url)}"
    
    # Check for repetitive loops
    if not error_integration.prevent_repetitive_loops(operation_name, max_retries):
        return {
            "success": False,
            "error": "Repetitive loop detected for web request",
            "suggestion": "Try a different URL or approach"
        }
    
    # Check cache first
    cached_result = error_handler.get_cached_result(operation_name)
    if cached_result:
        return cached_result
    
    # Perform actual request with error handling
    try:
        result = perform_actual_request(url)
        error_handler.cache_result(operation_name, result, ttl=300.0)
        error_integration.reset_operation_counter(operation_name)
        return result
    except Exception as e:
        return handle_error(e, operation_name)
```

## Configuration Options

### RetryConfig

```python
@dataclass
class RetryConfig:
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_backoff: bool = True
    jitter: bool = True
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 60.0
```

### Custom Configuration

```python
# Custom retry configuration for specific operations
custom_config = RetryConfig(
    max_attempts=5,
    base_delay=2.0,
    exponential_backoff=True,
    circuit_breaker_threshold=3
)

@with_error_handling("custom_operation", custom_config)
async def custom_operation():
    # Your operation here
    pass
```

## Error Handling Strategies

### 1. Network Errors
- **Retry**: Yes, with exponential backoff
- **Circuit Breaker**: Opens after multiple failures
- **Suggestion**: Check internet connection

### 2. Timeout Errors
- **Retry**: Yes, with increasing delays
- **Circuit Breaker**: Opens after threshold
- **Suggestion**: Try again later

### 3. Rate Limit Errors
- **Retry**: Yes, with exponential backoff
- **Circuit Breaker**: Stays closed
- **Suggestion**: Wait before retrying

### 4. Authentication Errors
- **Retry**: No
- **Circuit Breaker**: Opens immediately
- **Suggestion**: Check credentials

### 5. Resource Not Found (404)
- **Retry**: No
- **Circuit Breaker**: Opens immediately
- **Suggestion**: Check URL/resource path

## Monitoring and Status

### Get Error Handler Status

```python
def get_error_handling_status() -> Dict[str, Any]:
    """Get comprehensive error handling status."""
    return {
        "error_integration": error_integration.get_operation_status(),
        "active_operations": len(error_integration.operation_history),
        "circuit_breakers": error_handler.get_status()
    }
```

### Example Status Output

```json
{
    "error_integration": {
        "operation_history": {
            "web_request_abc123": 2,
            "api_request_xyz789": 1
        },
        "error_handler_status": {
            "circuit_breakers": {
                "web_request": {
                    "state": "CLOSED",
                    "failure_count": 0,
                    "last_failure_time": 0
                }
            },
            "loop_detector": {
                "active_operations": 2,
                "cached_results": 5
            }
        }
    },
    "active_operations": 2,
    "circuit_breakers": {
        "web_request": {
            "state": "CLOSED",
            "failure_count": 0
        }
    }
}
```

## Best Practices

### 1. Always Use Error Handling Decorators

```python
# Good
@with_error_handling("operation_name")
async def my_operation():
    # Your code here
    pass

# Bad
async def my_operation():
    # No error handling
    pass
```

### 2. Provide Meaningful Operation Names

```python
# Good
@with_error_handling("text_sentiment_analysis")

# Bad
@with_error_handling("op1")
```

### 3. Use Appropriate Retry Configurations

```python
# For critical operations
critical_config = RetryConfig(max_attempts=5, base_delay=2.0)

# For non-critical operations
simple_config = RetryConfig(max_attempts=2, base_delay=1.0)
```

### 4. Monitor Circuit Breaker States

```python
status = error_handler.get_status()
for operation, state in status['circuit_breakers'].items():
    if state['state'] == 'OPEN':
        logger.warning(f"Circuit breaker OPEN for {operation}")
```

### 5. Clear Cache When Needed

```python
# Clear all cached results
error_handler.clear_cache()

# Clear specific operation
error_integration.reset_operation_counter("operation_name")
```

## Integration Examples

### MCP Tool Integration

```python
def create_safe_mcp_tool(tool_name: str, max_retries: int = 3):
    """Create a safe MCP tool with error handling."""
    
    def decorator(func):
        @with_error_handling(tool_name)
        async def safe_wrapper(*args, **kwargs):
            operation_name = f"{tool_name}_{hash(str(args) + str(kwargs))}"
            
            # Check for repetitive loops
            if not error_integration.prevent_repetitive_loops(operation_name, max_retries):
                return {
                    "success": False,
                    "error": f"Repetitive loop detected for {tool_name}",
                    "suggestion": "Try different parameters or approach"
                }
            
            try:
                result = await func(*args, **kwargs)
                error_integration.reset_operation_counter(operation_name)
                return result
            except Exception as e:
                logger.error(f"{tool_name} failed: {str(e)}")
                return {
                    "success": False,
                    "error": str(e),
                    "tool": tool_name
                }
        
        return safe_wrapper
    
    return decorator

# Usage
safe_tool = create_safe_mcp_tool("my_tool", max_retries=3)
```

### Web Request Integration

```python
def fetch_with_error_handling(url: str) -> Dict[str, Any]:
    """Fetch web content with comprehensive error handling."""
    operation_name = f"fetch_{hash(url)}"
    
    # Check for repetitive loops
    if not error_integration.prevent_repetitive_loops(operation_name):
        return {
            "success": False,
            "error": "Repetitive loop detected",
            "suggestion": "Try a different approach"
        }
    
    # Check cache
    cached = error_handler.get_cached_result(operation_name)
    if cached:
        return cached
    
    # Perform fetch with error handling
    try:
        result = perform_actual_fetch(url)
        error_handler.cache_result(operation_name, result)
        error_integration.reset_operation_counter(operation_name)
        return result
    except Exception as e:
        return handle_fetch_error(e, operation_name)
```

## Testing Error Handling

### Test Circuit Breaker

```python
async def test_circuit_breaker():
    """Test circuit breaker functionality."""
    
    @with_error_handling("test_operation")
    async def failing_operation():
        raise Exception("Simulated failure")
    
    # First few attempts should fail but retry
    for i in range(3):
        try:
            await failing_operation()
        except Exception as e:
            print(f"Attempt {i+1} failed: {e}")
    
    # After threshold, circuit should be open
    status = error_handler.get_status()
    circuit_state = status['circuit_breakers']['test_operation']['state']
    print(f"Circuit state: {circuit_state}")
```

### Test Loop Detection

```python
def test_loop_detection():
    """Test repetitive loop detection."""
    
    operation_name = "test_loop"
    
    # Simulate repetitive attempts
    for i in range(10):
        should_continue = error_integration.prevent_repetitive_loops(operation_name, max_attempts=5)
        print(f"Attempt {i+1}: should_continue = {should_continue}")
        
        if not should_continue:
            print("Loop detection triggered!")
            break
```

## Conclusion

The error handling system provides:

1. **Repetitive Loop Prevention**: Stops infinite retry loops
2. **Circuit Breaker Pattern**: Prevents cascading failures
3. **Intelligent Retry Logic**: Different strategies for different error types
4. **Caching**: Reduces redundant operations
5. **Monitoring**: Visibility into system health
6. **Easy Integration**: Simple decorators for existing code

This system ensures the application remains stable and responsive even when external services fail or network issues occur.
