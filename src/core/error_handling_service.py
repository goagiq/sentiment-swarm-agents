"""
Error Handling Service for consistent error handling across agents.
Provides standardized error handling, logging, and recovery mechanisms.
"""

import asyncio
import traceback
from typing import Any, Callable, Dict, Optional, Type
from functools import wraps
from enum import Enum

import logging

# Configure logger
logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorType(Enum):
    """Error types for categorization."""
    VALIDATION = "validation"
    PROCESSING = "processing"
    NETWORK = "network"
    RESOURCE = "resource"
    CONFIGURATION = "configuration"
    UNKNOWN = "unknown"


class ErrorContext:
    """Context information for error handling."""

    def __init__(self, agent_id: str, operation: str, **kwargs):
        self.agent_id = agent_id
        self.operation = operation
        self.additional_info = kwargs
        self.timestamp = asyncio.get_event_loop().time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'agent_id': self.agent_id,
            'operation': self.operation,
            'timestamp': self.timestamp,
            **self.additional_info
        }


class ErrorInfo:
    """Information about an error."""

    def __init__(
        self,
        error: Exception,
        context: ErrorContext,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        error_type: ErrorType = ErrorType.UNKNOWN
    ):
        self.error = error
        self.context = context
        self.severity = severity
        self.error_type = error_type
        self.traceback = traceback.format_exc()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'error_message': str(self.error),
            'error_type': self.error_type.value,
            'severity': self.severity.value,
            'context': self.context.to_dict(),
            'traceback': self.traceback
        }


class ErrorHandlingService:
    """Service for consistent error handling across agents."""

    def __init__(self):
        self.logger = logger
        self.error_handlers: Dict[Type[Exception], Callable] = {}
        self.recovery_strategies: Dict[ErrorType, Callable] = {}
        self.error_stats = {
            'total_errors': 0,
            'errors_by_type': {},
            'errors_by_severity': {},
            'recovery_attempts': 0,
            'successful_recoveries': 0
        }

    def register_error_handler(
        self,
        exception_type: Type[Exception],
        handler: Callable
    ):
        """Register a custom error handler for a specific exception type."""
        self.error_handlers[exception_type] = handler

    def register_recovery_strategy(
        self,
        error_type: ErrorType,
        strategy: Callable
    ):
        """Register a recovery strategy for an error type."""
        self.recovery_strategies[error_type] = strategy

    def handle_error(
        self,
        error: Exception,
        context: ErrorContext,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        error_type: ErrorType = ErrorType.UNKNOWN
    ) -> ErrorInfo:
        """Handle an error with consistent logging and categorization."""
        error_info = ErrorInfo(error, context, severity, error_type)

        # Update statistics
        self.error_stats['total_errors'] += 1
        self.error_stats['errors_by_type'][error_type.value] = (
            self.error_stats['errors_by_type'].get(error_type.value, 0) + 1
        )
        self.error_stats['errors_by_severity'][severity.value] = (
            self.error_stats['errors_by_severity'].get(severity.value, 0) + 1
        )

        # Log based on severity
        log_message = (
            f"Error in {context.agent_id} during {context.operation}: {error}"
        )

        if severity == ErrorSeverity.LOW:
            self.logger.warning(log_message)
        elif severity == ErrorSeverity.MEDIUM:
            self.logger.error(log_message)
        elif severity == ErrorSeverity.HIGH:
            self.logger.error(log_message)
            self.logger.error(f"Traceback: {error_info.traceback}")
        elif severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message)
            self.logger.critical(f"Traceback: {error_info.traceback}")

        # Try custom handler
        for exception_type, handler in self.error_handlers.items():
            if isinstance(error, exception_type):
                try:
                    handler(error_info)
                except Exception as handler_error:
                    self.logger.error(f"Error in custom handler: {handler_error}")

        return error_info

    async def attempt_recovery(
        self,
        error_info: ErrorInfo,
        recovery_func: Optional[Callable] = None
    ) -> bool:
        """Attempt to recover from an error."""
        self.error_stats['recovery_attempts'] += 1

        try:
            if recovery_func:
                # Use provided recovery function
                if asyncio.iscoroutinefunction(recovery_func):
                    await recovery_func(error_info)
                else:
                    recovery_func(error_info)
            elif error_info.error_type in self.recovery_strategies:
                # Use registered recovery strategy
                strategy = self.recovery_strategies[error_info.error_type]
                if asyncio.iscoroutinefunction(strategy):
                    await strategy(error_info)
                else:
                    strategy(error_info)
            else:
                # Default recovery attempt
                await self._default_recovery(error_info)

            self.error_stats['successful_recoveries'] += 1
            self.logger.info(
                f"Successfully recovered from error in {error_info.context.agent_id}"
            )
            return True

        except Exception as recovery_error:
            self.logger.error(f"Recovery failed: {recovery_error}")
            return False

    async def _default_recovery(self, error_info: ErrorInfo):
        """Default recovery strategy."""
        # Wait a bit before retrying
        await asyncio.sleep(1)

        # For network errors, try again
        if error_info.error_type == ErrorType.NETWORK:
            self.logger.info("Attempting network error recovery...")

        # For resource errors, try to free up resources
        elif error_info.error_type == ErrorType.RESOURCE:
            self.logger.info("Attempting resource cleanup...")
            # Could implement memory cleanup, connection reset, etc.

    def with_error_handling(
        self,
        context: ErrorContext,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        error_type: ErrorType = ErrorType.UNKNOWN,
        recovery_func: Optional[Callable] = None
    ):
        """Decorator for automatic error handling."""
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except Exception as error:
                    error_info = self.handle_error(error, context, severity, error_type)
                    await self.attempt_recovery(error_info, recovery_func)
                    raise

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as error:
                    error_info = self.handle_error(error, context, severity, error_type)
                    # For sync functions, we can't await recovery
                    asyncio.create_task(
                        self.attempt_recovery(error_info, recovery_func)
                    )
                    raise

            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper

        return decorator

    def validate_input(
        self,
        value: Any,
        validation_func: Callable[[Any], bool],
        error_message: str,
        context: ErrorContext
    ) -> bool:
        """Validate input with error handling."""
        try:
            if not validation_func(value):
                raise ValueError(error_message)
            return True
        except Exception as error:
            self.handle_error(
                error,
                context,
                ErrorSeverity.LOW,
                ErrorType.VALIDATION
            )
            return False

    def safe_execute(
        self,
        func: Callable,
        *args,
        context: ErrorContext,
        default_return: Any = None,
        **kwargs
    ) -> Any:
        """Safely execute a function with error handling."""
        try:
            return func(*args, **kwargs)
        except Exception as error:
            self.handle_error(error, context)
            return default_return

    async def safe_execute_async(
        self,
        func: Callable,
        *args,
        context: ErrorContext,
        default_return: Any = None,
        **kwargs
    ) -> Any:
        """Safely execute an async function with error handling."""
        try:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        except Exception as error:
            self.handle_error(error, context)
            return default_return

    def get_error_stats(self) -> Dict[str, Any]:
        """Get error handling statistics."""
        return {
            **self.error_stats,
            'recovery_rate': (
                self.error_stats['successful_recoveries']
                / max(self.error_stats['recovery_attempts'], 1) * 100
            )
        }

    def reset_stats(self):
        """Reset error statistics."""
        self.error_stats = {
            'total_errors': 0,
            'errors_by_type': {},
            'errors_by_severity': {},
            'recovery_attempts': 0,
            'successful_recoveries': 0
        }


# Global instance
error_handling_service = ErrorHandlingService()
