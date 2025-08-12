"""
Lazy loading system for deferring heavy initializations.
This module provides utilities to improve startup performance by loading
components only when they're actually needed.
"""

import asyncio
import threading
from typing import Any, Callable, Dict, Optional
from loguru import logger


class LazyLoader:
    """Generic lazy loader for deferring object creation."""
    
    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._factories: Dict[str, Callable] = {}
        self._initializing: Dict[str, bool] = {}
    
    def register(self, key: str, factory_func: Callable) -> None:
        """Register a factory function for lazy creation."""
        self._factories[key] = factory_func
        logger.debug(f"Registered lazy loader for: {key}")
    
    def get(self, key: str) -> Any:
        """Get an object, creating it if necessary."""
        if key not in self._cache:
            if key not in self._factories:
                raise KeyError(f"No factory registered for key: {key}")
            
            if self._initializing.get(key, False):
                raise RuntimeError(f"Circular dependency detected for key: {key}")
            
            self._initializing[key] = True
            try:
                logger.debug(f"Lazy loading: {key}")
                self._cache[key] = self._factories[key]()
                logger.info(f"✅ Lazy loaded: {key}")
            finally:
                self._initializing[key] = False
        
        return self._cache[key]
    
    def get_or_none(self, key: str) -> Optional[Any]:
        """Get an object if it exists, otherwise return None."""
        try:
            return self.get(key)
        except (KeyError, RuntimeError):
            return None
    
    def is_loaded(self, key: str) -> bool:
        """Check if an object has been loaded."""
        return key in self._cache
    
    def preload(self, key: str) -> None:
        """Preload an object without returning it."""
        self.get(key)
    
    def clear(self, key: str = None) -> None:
        """Clear cached objects."""
        if key:
            self._cache.pop(key, None)
            logger.debug(f"Cleared cache for: {key}")
        else:
            self._cache.clear()
            logger.debug("Cleared all cached objects")


class AsyncLazyLoader:
    """Async lazy loader for deferring async object creation."""
    
    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._factories: Dict[str, Callable] = {}
        self._initializing: Dict[str, asyncio.Task] = {}
    
    def register(self, key: str, factory_func: Callable) -> None:
        """Register an async factory function for lazy creation."""
        self._factories[key] = factory_func
        logger.debug(f"Registered async lazy loader for: {key}")
    
    async def get(self, key: str) -> Any:
        """Get an object asynchronously, creating it if necessary."""
        if key not in self._cache:
            if key not in self._factories:
                raise KeyError(f"No factory registered for key: {key}")
            
            if key in self._initializing:
                # Wait for existing initialization
                await self._initializing[key]
                return self._cache[key]
            
            # Start new initialization
            task = asyncio.create_task(self._initialize_object(key))
            self._initializing[key] = task
            
            try:
                await task
            finally:
                self._initializing.pop(key, None)
        
        return self._cache[key]
    
    async def _initialize_object(self, key: str) -> None:
        """Initialize an object asynchronously."""
        try:
            logger.debug(f"Async lazy loading: {key}")
            self._cache[key] = await self._factories[key]()
            logger.info(f"✅ Async lazy loaded: {key}")
        except Exception as e:
            logger.error(f"Failed to async lazy load {key}: {e}")
            raise
    
    async def get_or_none(self, key: str) -> Optional[Any]:
        """Get an object if it exists, otherwise return None."""
        try:
            return await self.get(key)
        except (KeyError, RuntimeError):
            return None
    
    def is_loaded(self, key: str) -> bool:
        """Check if an object has been loaded."""
        return key in self._cache
    
    async def preload(self, key: str) -> None:
        """Preload an object without returning it."""
        await self.get(key)
    
    def clear(self, key: str = None) -> None:
        """Clear cached objects."""
        if key:
            self._cache.pop(key, None)
            logger.debug(f"Cleared async cache for: {key}")
        else:
            self._cache.clear()
            logger.debug("Cleared all async cached objects")


class ServiceManager:
    """Service manager for coordinating lazy-loaded services."""
    
    def __init__(self):
        self._sync_loader = LazyLoader()
        self._async_loader = AsyncLazyLoader()
        self._initialization_tasks: Dict[str, asyncio.Task] = {}
    
    def register_sync_service(self, name: str, factory_func: Callable) -> None:
        """Register a synchronous service."""
        self._sync_loader.register(name, factory_func)
    
    def register_async_service(self, name: str, factory_func: Callable) -> None:
        """Register an asynchronous service."""
        self._async_loader.register(name, factory_func)
    
    def get_sync_service(self, name: str) -> Any:
        """Get a synchronous service."""
        return self._sync_loader.get(name)
    
    async def get_async_service(self, name: str) -> Any:
        """Get an asynchronous service."""
        return await self._async_loader.get(name)
    
    def start_background_init(self, service_names: list) -> None:
        """Start background initialization of services."""
        for service_name in service_names:
            if service_name not in self._initialization_tasks:
                # Create a thread to run the async task
                def run_async_init(service_name=service_name):
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        loop.run_until_complete(self._background_init_service(service_name))
                    except Exception as e:
                        logger.error(f"Background initialization failed for {service_name}: {e}")
                
                thread = threading.Thread(target=run_async_init, daemon=True)
                thread.start()
                self._initialization_tasks[service_name] = thread
                logger.info(f"Started background initialization for: {service_name}")
    
    async def _background_init_service(self, service_name: str) -> None:
        """Initialize a service in the background."""
        try:
            if self._sync_loader.is_loaded(service_name):
                return
            
            if self._async_loader.is_loaded(service_name):
                return
            
            # Try async first, then sync
            try:
                await self._async_loader.get(service_name)
            except KeyError:
                self._sync_loader.get(service_name)
            
            logger.info(f"✅ Background initialization completed for: {service_name}")
        except Exception as e:
            logger.error(f"Background initialization failed for {service_name}: {e}")
    
    async def wait_for_services(self, service_names: list, timeout: float = 30.0) -> None:
        """Wait for services to be initialized."""
        tasks = []
        for service_name in service_names:
            if service_name in self._initialization_tasks:
                tasks.append(self._initialization_tasks[service_name])
        
        if tasks:
            try:
                await asyncio.wait_for(asyncio.gather(*tasks), timeout=timeout)
                logger.info(f"All services initialized: {service_names}")
            except asyncio.TimeoutError:
                logger.warning(f"Timeout waiting for services: {service_names}")
    
    def get_initialization_status(self) -> Dict[str, str]:
        """Get the initialization status of all services."""
        status = {}
        
        # Check sync services
        for key in self._sync_loader._factories:
            if self._sync_loader.is_loaded(key):
                status[key] = "loaded"
            elif key in self._initialization_tasks:
                status[key] = "initializing"
            else:
                status[key] = "pending"
        
        # Check async services
        for key in self._async_loader._factories:
            if self._async_loader.is_loaded(key):
                status[key] = "loaded"
            elif key in self._initialization_tasks:
                status[key] = "initializing"
            else:
                status[key] = "pending"
        
        return status


# Global service manager instance
service_manager = ServiceManager()
