"""
Integration module for error handling system.
Shows how to properly integrate error handling to prevent repetitive loops.
"""

import asyncio
import logging
from typing import Dict, Any
from .error_handler import (
    error_handler, 
    with_error_handling, 
    ErrorType
)

logger = logging.getLogger(__name__)


class ErrorHandlingIntegration:
    """Integration class for error handling in the main application."""
    
    def __init__(self):
        self.error_handler = error_handler
        self.operation_history: Dict[str, int] = {}
    
    def prevent_repetitive_loops(self, operation_name: str, max_attempts: int = 3) -> bool:
        """
        Prevent repetitive loops by tracking operation attempts.
        
        Args:
            operation_name: Name of the operation
            max_attempts: Maximum number of attempts allowed
            
        Returns:
            bool: True if operation should continue, False if loop detected
        """
        if operation_name not in self.operation_history:
            self.operation_history[operation_name] = 0
        
        self.operation_history[operation_name] += 1
        
        if self.operation_history[operation_name] > max_attempts:
            logger.warning(f"Repetitive loop detected for {operation_name}. Stopping execution.")
            return False
        
        return True
    
    def reset_operation_counter(self, operation_name: str):
        """Reset the attempt counter for an operation."""
        if operation_name in self.operation_history:
            self.operation_history[operation_name] = 0
    
    def get_operation_status(self) -> Dict[str, Any]:
        """Get status of all operations."""
        return {
            'operation_history': self.operation_history,
            'error_handler_status': self.error_handler.get_status()
        }


# Global integration instance
error_integration = ErrorHandlingIntegration()


def safe_web_request(url: str, max_retries: int = 3) -> Dict[str, Any]:
    """
    Safe web request function with repetitive loop prevention.
    
    This function demonstrates how to prevent the repetitive loop issue
    that occurred in the original implementation.
    """
    operation_name = f"web_request_{hash(url)}"
    
    # Check for repetitive loops
    if not error_integration.prevent_repetitive_loops(operation_name, max_retries):
        return {
            "success": False,
            "error": "Repetitive loop detected for web request",
            "suggestion": "Try a different URL or approach",
            "operation": operation_name
        }
    
    # Check cache first
    cached_result = error_handler.get_cached_result(operation_name)
    if cached_result:
        logger.info(f"Using cached result for {operation_name}")
        return cached_result
    
    try:
        # Simulate web request (replace with actual implementation)
        logger.info(f"Making web request to: {url}")
        
        # Here you would implement the actual web request logic
        # For now, we'll simulate a successful request
        result = {
            "success": True,
            "url": url,
            "content": f"Content from {url}",
            "operation": operation_name
        }
        
        # Cache the result
        error_handler.cache_result(operation_name, result, ttl=300.0)
        
        # Reset operation counter on success
        error_integration.reset_operation_counter(operation_name)
        
        return result
        
    except Exception as e:
        error_type = error_handler.classify_error(e)
        logger.error(f"Web request failed: {str(e)} (Type: {error_type.value})")
        
        return {
            "success": False,
            "error": str(e),
            "error_type": error_type.value,
            "operation": operation_name,
            "suggestion": _get_suggestion_for_error(error_type)
        }


def _get_suggestion_for_error(error_type: ErrorType) -> str:
    """Get suggestion based on error type."""
    suggestions = {
        ErrorType.NETWORK: "Check your internet connection and try again",
        ErrorType.TIMEOUT: "The request timed out. Try again later",
        ErrorType.RATE_LIMIT: "Rate limit exceeded. Wait before retrying",
        ErrorType.AUTHENTICATION: "Authentication failed. Check credentials",
        ErrorType.VALIDATION: "Invalid request. Check input parameters",
        ErrorType.RESOURCE_NOT_FOUND: "Resource not found. Check the URL",
        ErrorType.SERVER_ERROR: "Server error. Try again later",
        ErrorType.UNKNOWN: "Unknown error occurred. Try again later"
    }
    return suggestions.get(error_type, "Try again later")


@with_error_handling("api_request")
async def safe_api_request(endpoint: str, data: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Safe API request with error handling and loop prevention.
    
    Args:
        endpoint: API endpoint URL
        data: Request data
        
    Returns:
        Dict containing the response or error information
    """
    operation_name = f"api_request_{endpoint}"
    
    # Check for repetitive loops
    if not error_integration.prevent_repetitive_loops(operation_name):
        return {
            "success": False,
            "error": "Repetitive loop detected for API request",
            "endpoint": endpoint
        }
    
    try:
        # Simulate API request (replace with actual implementation)
        logger.info(f"Making API request to: {endpoint}")
        
        # Here you would implement the actual API request logic
        result = {
            "success": True,
            "endpoint": endpoint,
            "data": data,
            "response": f"Response from {endpoint}"
        }
        
        # Reset operation counter on success
        error_integration.reset_operation_counter(operation_name)
        
        return result
        
    except Exception as e:
        logger.error(f"API request failed: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "endpoint": endpoint
        }


def integrate_with_mcp_tools():
    """
    Example of how to integrate error handling with MCP tools.
    
    This function shows the pattern for preventing repetitive loops
    in MCP tool implementations.
    """
    
    def create_safe_mcp_tool(tool_name: str, max_retries: int = 3):
        """Create a safe MCP tool with error handling."""
        
        def decorator(func):
            @with_error_handling(tool_name)
            async def safe_wrapper(*args, **kwargs):
                operation_name = f"{tool_name}_{hash(str(args) + str(kwargs))}"
                
                # Check for repetitive loops
                if not error_integration.prevent_repetitive_loops(
                    operation_name, max_retries
                ):
                    return {
                        "success": False,
                        "error": f"Repetitive loop detected for {tool_name}",
                        "suggestion": "Try different parameters or approach"
                    }
                
                try:
                    result = await func(*args, **kwargs)
                    
                    # Reset operation counter on success
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
    
    return create_safe_mcp_tool


def get_error_handling_status() -> Dict[str, Any]:
    """Get comprehensive error handling status."""
    return {
        "error_integration": error_integration.get_operation_status(),
        "active_operations": len(error_integration.operation_history),
        "circuit_breakers": error_handler.get_status()
    }


# Example usage functions
async def example_usage():
    """Example of how to use the error handling system."""
    
    # Example 1: Safe web request
    result1 = safe_web_request("https://example.com")
    print(f"Web request result: {result1}")
    
    # Example 2: Safe API request
    result2 = await safe_api_request("/api/data", {"key": "value"})
    print(f"API request result: {result2}")
    
    # Example 3: Get status
    status = get_error_handling_status()
    print(f"Error handling status: {status}")


if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())
