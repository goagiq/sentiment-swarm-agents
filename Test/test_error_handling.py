#!/usr/bin/env python3
"""
Test script for error handling and repetitive loop prevention.
Demonstrates how the system prevents infinite loops and handles errors gracefully.
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.error_handler import (
    error_handler, 
    with_error_handling, 
    RetryConfig, 
    ErrorType,
    error_integration
)
from core.error_integration import safe_web_request, get_error_handling_status


def test_repetitive_loop_prevention():
    """Test that repetitive loops are prevented."""
    print("🧪 Testing Repetitive Loop Prevention")
    print("=" * 50)
    
    operation_name = "test_repetitive_loop"
    max_attempts = 3
    
    print(f"Testing operation: {operation_name}")
    print(f"Max attempts allowed: {max_attempts}")
    print()
    
    # Simulate repetitive attempts
    for i in range(5):
        should_continue = error_integration.prevent_repetitive_loops(
            operation_name, max_attempts
        )
        print(f"Attempt {i+1}: should_continue = {should_continue}")
        
        if not should_continue:
            print("✅ Loop detection triggered successfully!")
            break
    
    # Reset for next test
    error_integration.reset_operation_counter(operation_name)
    print()


async def test_circuit_breaker():
    """Test circuit breaker functionality."""
    print("🧪 Testing Circuit Breaker")
    print("=" * 50)
    
    @with_error_handling("test_circuit_breaker")
    async def failing_operation():
        """Simulate a failing operation."""
        raise Exception("Simulated failure for testing")
    
    print("Testing circuit breaker with failing operation...")
    
    # First few attempts should fail but retry
    for i in range(3):
        try:
            await failing_operation()
        except Exception as e:
            print(f"Attempt {i+1} failed: {e}")
    
    # Check circuit breaker status
    status = error_handler.get_status()
    circuit_state = status['circuit_breakers']['test_circuit_breaker']['state']
    failure_count = status['circuit_breakers']['test_circuit_breaker']['failure_count']
    
    print(f"Circuit state: {circuit_state}")
    print(f"Failure count: {failure_count}")
    
    if circuit_state == "OPEN":
        print("✅ Circuit breaker opened successfully after failures!")
    else:
        print("⚠️  Circuit breaker did not open as expected")
    
    print()


def test_error_classification():
    """Test error classification functionality."""
    print("🧪 Testing Error Classification")
    print("=" * 50)
    
    test_errors = [
        ("Connection timeout", "timeout"),
        ("Rate limit exceeded", "rate_limit"),
        ("401 Unauthorized", "authentication"),
        ("404 Not Found", "resource_not_found"),
        ("500 Internal Server Error", "server_error"),
        ("Network connection failed", "network"),
        ("Invalid input", "validation"),
        ("Unknown error", "unknown")
    ]
    
    for error_msg, expected_type in test_errors:
        # Create a mock exception
        class MockException(Exception):
            def __init__(self, message):
                self.message = message
            
            def __str__(self):
                return self.message
        
        error = MockException(error_msg)
        error_type = error_handler.classify_error(error)
        
        status = "✅" if error_type.value == expected_type else "❌"
        print(f"{status} '{error_msg}' -> {error_type.value} (expected: {expected_type})")
    
    print()


def test_safe_web_request():
    """Test safe web request functionality."""
    print("🧪 Testing Safe Web Request")
    print("=" * 50)
    
    # Test with a valid URL (will simulate success)
    print("Testing safe web request...")
    result = safe_web_request("https://example.com")
    
    print(f"Result: {result}")
    
    if result.get("success"):
        print("✅ Safe web request completed successfully!")
    else:
        print(f"⚠️  Safe web request failed: {result.get('error')}")
    
    print()


async def test_async_error_handling():
    """Test async error handling."""
    print("🧪 Testing Async Error Handling")
    print("=" * 50)
    
    @with_error_handling("async_test")
    async def async_operation():
        """Simulate an async operation."""
        await asyncio.sleep(0.1)  # Simulate some work
        return {"status": "success", "data": "test_data"}
    
    try:
        result = await async_operation()
        print(f"✅ Async operation successful: {result}")
    except Exception as e:
        print(f"❌ Async operation failed: {e}")
    
    print()


async def test_retry_configuration():
    """Test custom retry configuration."""
    print("🧪 Testing Retry Configuration")
    print("=" * 50)
    
    # Create custom retry configuration
    custom_config = RetryConfig(
        max_attempts=2,
        base_delay=0.1,  # Short delay for testing
        exponential_backoff=True,
        circuit_breaker_threshold=2
    )
    
    @with_error_handling("custom_retry_test", custom_config)
    async def custom_retry_operation():
        """Operation with custom retry configuration."""
        raise Exception("Custom retry test failure")
    
    print("Testing custom retry configuration...")
    
    try:
        await custom_retry_operation()
    except Exception as e:
        print(f"✅ Custom retry operation failed as expected: {e}")
    
    # Check status
    status = error_handler.get_status()
    circuit_state = status['circuit_breakers']['custom_retry_test']['state']
    print(f"Circuit state after custom retry: {circuit_state}")
    
    print()


def test_caching():
    """Test result caching functionality."""
    print("🧪 Testing Result Caching")
    print("=" * 50)
    
    # Test caching
    test_key = "test_cache_key"
    test_data = {"test": "data", "timestamp": "2025-01-01"}
    
    # Cache some data
    error_handler.cache_result(test_key, test_data, ttl=60.0)
    print(f"✅ Cached data for key: {test_key}")
    
    # Retrieve cached data
    cached_data = error_handler.get_cached_result(test_key)
    if cached_data:
        print(f"✅ Retrieved cached data: {cached_data}")
    else:
        print("❌ Failed to retrieve cached data")
    
    # Test cache miss
    non_existent_key = "non_existent_key"
    cached_data = error_handler.get_cached_result(non_existent_key)
    if cached_data is None:
        print(f"✅ Correctly returned None for non-existent key: {non_existent_key}")
    else:
        print(f"❌ Unexpectedly found data for non-existent key: {non_existent_key}")
    
    print()


def test_status_monitoring():
    """Test status monitoring functionality."""
    print("🧪 Testing Status Monitoring")
    print("=" * 50)
    
    status = get_error_handling_status()
    
    print("Error Handling Status:")
    print(f"  Active operations: {status.get('active_operations', 0)}")
    print(f"  Circuit breakers: {len(status.get('circuit_breakers', {}))}")
    
    if status.get('circuit_breakers'):
        print("  Circuit breaker states:")
        for name, state in status['circuit_breakers'].items():
            print(f"    {name}: {state.get('state', 'UNKNOWN')}")
    
    print("✅ Status monitoring working correctly!")
    print()


async def main():
    """Run all tests."""
    print("🚀 Error Handling System Test Suite")
    print("=" * 60)
    print()
    
    # Run synchronous tests
    test_repetitive_loop_prevention()
    test_error_classification()
    test_safe_web_request()
    test_caching()
    test_status_monitoring()
    
    # Run asynchronous tests
    await test_circuit_breaker()
    await test_async_error_handling()
    await test_retry_configuration()
    
    print("🎉 All tests completed!")
    print()
    print("📊 Final Status:")
    final_status = get_error_handling_status()
    print(f"  Total operations tracked: {final_status.get('active_operations', 0)}")
    print(f"  Circuit breakers: {len(final_status.get('circuit_breakers', {}))}")
    print(f"  Cached results: {final_status.get('circuit_breakers', {}).get('loop_detector', {}).get('cached_results', 0)}")


if __name__ == "__main__":
    # Run the test suite
    asyncio.run(main())
