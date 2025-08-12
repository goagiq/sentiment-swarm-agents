"""
Memory management service for multilingual content processing.
Implements memory monitoring, cleanup, and optimization strategies.
"""

import psutil
import gc
import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class MemoryThreshold:
    """Memory threshold configuration."""
    warning_threshold: float = 0.7  # 70% of max memory
    critical_threshold: float = 0.85  # 85% of max memory
    cleanup_threshold: float = 0.6  # 60% triggers cleanup


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    total_memory: int
    available_memory: int
    used_memory: int
    memory_percentage: float
    timestamp: datetime


class MemoryManager:
    """Memory management system for multilingual processing."""
    
    def __init__(self, 
                 max_memory_mb: int = 1024,
                 cleanup_interval_seconds: int = 300,
                 enable_auto_cleanup: bool = True):
        self.max_memory_mb = max_memory_mb
        self.cleanup_interval_seconds = cleanup_interval_seconds
        self.enable_auto_cleanup = enable_auto_cleanup
        
        # Memory thresholds
        self.thresholds = MemoryThreshold()
        
        # Memory tracking
        self.memory_history: List[MemoryStats] = []
        self.max_history_size = 100
        
        # Cleanup tracking
        self.last_cleanup_time = time.time()
        self.cleanup_count = 0
        
        # Memory-intensive objects tracking
        self.tracked_objects: Dict[str, Any] = {}
        self.object_sizes: Dict[str, int] = {}
        
        # Performance metrics
        self.stats = {
            "cleanups_performed": 0,
            "total_memory_freed_mb": 0.0,
            "peak_memory_usage_mb": 0.0,
            "average_memory_usage_mb": 0.0,
            "memory_warnings": 0,
            "critical_memory_events": 0
        }
        
        # Start monitoring if auto cleanup is enabled
        if self.enable_auto_cleanup:
            try:
                asyncio.create_task(self._monitor_memory())
            except RuntimeError:
                # No running event loop, will start monitoring when needed
                pass
    
    async def check_memory_usage(self) -> Dict[str, Any]:
        """Check current memory usage and trigger cleanup if needed."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            current_mb = memory_info.rss / 1024 / 1024
            
            # Get system memory info
            system_memory = psutil.virtual_memory()
            
            # Create memory stats
            memory_stats = MemoryStats(
                total_memory=system_memory.total,
                available_memory=system_memory.available,
                used_memory=system_memory.used,
                memory_percentage=system_memory.percent / 100.0,
                timestamp=datetime.now()
            )
            
            # Update history
            self._update_memory_history(memory_stats)
            
            # Update peak memory usage
            self.stats["peak_memory_usage_mb"] = max(
                self.stats["peak_memory_usage_mb"], 
                current_mb
            )
            
            # Check thresholds
            memory_usage_ratio = current_mb / self.max_memory_mb
            
            if memory_usage_ratio > self.thresholds.critical_threshold:
                self.stats["critical_memory_events"] += 1
                logger.warning(f"Critical memory usage: {current_mb:.1f}MB ({memory_usage_ratio:.1%})")
                await self.emergency_cleanup()
                
            elif memory_usage_ratio > self.thresholds.warning_threshold:
                self.stats["memory_warnings"] += 1
                logger.warning(f"High memory usage: {current_mb:.1f}MB ({memory_usage_ratio:.1%})")
                
                # Trigger cleanup if enough time has passed
                if time.time() - self.last_cleanup_time > self.cleanup_interval_seconds:
                    await self.cleanup_memory()
            
            return {
                "current_memory_mb": current_mb,
                "memory_usage_ratio": memory_usage_ratio,
                "system_memory_percentage": memory_stats.memory_percentage,
                "available_system_memory_mb": memory_stats.available_memory / 1024 / 1024,
                "threshold_status": self._get_threshold_status(memory_usage_ratio)
            }
            
        except Exception as e:
            logger.error(f"Error checking memory usage: {e}")
            return {"error": str(e)}
    
    async def cleanup_memory(self) -> Dict[str, Any]:
        """Perform memory cleanup operations."""
        try:
            start_time = time.time()
            initial_memory = self._get_current_memory_mb()
            
            # Clear memory cache
            await self._clear_memory_cache()
            
            # Clear translation memory
            await self._clear_translation_memory()
            
            # Clear tracked objects
            await self._clear_tracked_objects()
            
            # Force garbage collection
            gc.collect()
            
            # Clear memory history if too large
            if len(self.memory_history) > self.max_history_size:
                self.memory_history = self.memory_history[-self.max_history_size//2:]
            
            # Calculate memory freed
            final_memory = self._get_current_memory_mb()
            memory_freed = initial_memory - final_memory
            
            # Update statistics
            self.stats["cleanups_performed"] += 1
            self.stats["total_memory_freed_mb"] += memory_freed
            self.last_cleanup_time = time.time()
            
            cleanup_time = time.time() - start_time
            
            logger.info(f"Memory cleanup completed: {memory_freed:.1f}MB freed in {cleanup_time:.2f}s")
            
            return {
                "memory_freed_mb": memory_freed,
                "cleanup_time_seconds": cleanup_time,
                "total_cleanups": self.stats["cleanups_performed"]
            }
            
        except Exception as e:
            logger.error(f"Error during memory cleanup: {e}")
            return {"error": str(e)}
    
    async def emergency_cleanup(self) -> Dict[str, Any]:
        """Perform emergency memory cleanup."""
        try:
            logger.warning("Performing emergency memory cleanup")
            
            # More aggressive cleanup
            result = await self.cleanup_memory()
            
            # Additional emergency measures
            await self._emergency_measures()
            
            return result
            
        except Exception as e:
            logger.error(f"Error during emergency cleanup: {e}")
            return {"error": str(e)}
    
    def track_object(self, name: str, obj: Any, estimated_size_mb: float = 0.0):
        """Track a memory-intensive object."""
        try:
            self.tracked_objects[name] = obj
            self.object_sizes[name] = estimated_size_mb
            
            logger.debug(f"Tracking object '{name}' with estimated size {estimated_size_mb:.1f}MB")
            
        except Exception as e:
            logger.error(f"Error tracking object '{name}': {e}")
    
    def untrack_object(self, name: str):
        """Stop tracking an object."""
        try:
            if name in self.tracked_objects:
                del self.tracked_objects[name]
                del self.object_sizes[name]
                logger.debug(f"Stopped tracking object '{name}'")
                
        except Exception as e:
            logger.error(f"Error untracking object '{name}': {e}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        try:
            current_memory = self._get_current_memory_mb()
            
            # Calculate average memory usage
            if self.memory_history:
                avg_memory = sum(
                    stats.used_memory / 1024 / 1024 
                    for stats in self.memory_history
                ) / len(self.memory_history)
            else:
                avg_memory = current_memory
            
            self.stats["average_memory_usage_mb"] = avg_memory
            
            return {
                "current_memory_mb": current_memory,
                "peak_memory_mb": self.stats["peak_memory_usage_mb"],
                "average_memory_mb": avg_memory,
                "max_memory_mb": self.max_memory_mb,
                "memory_usage_percentage": (current_memory / self.max_memory_mb) * 100,
                "cleanups_performed": self.stats["cleanups_performed"],
                "total_memory_freed_mb": self.stats["total_memory_freed_mb"],
                "memory_warnings": self.stats["memory_warnings"],
                "critical_events": self.stats["critical_memory_events"],
                "tracked_objects_count": len(self.tracked_objects),
                "tracked_objects_size_mb": sum(self.object_sizes.values()),
                "memory_history_size": len(self.memory_history),
                "last_cleanup_seconds_ago": time.time() - self.last_cleanup_time
            }
            
        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            return {"error": str(e)}
    
    async def optimize_for_language(self, language: str) -> Dict[str, Any]:
        """Optimize memory settings for specific language."""
        try:
            # Language-specific memory optimizations
            if language == "zh":  # Chinese: larger memory allocation
                self.max_memory_mb = int(self.max_memory_mb * 1.5)
                self.thresholds.warning_threshold = 0.75
                self.thresholds.critical_threshold = 0.9
                
            elif language == "ja":  # Japanese: moderate optimization
                self.max_memory_mb = int(self.max_memory_mb * 1.2)
                self.thresholds.warning_threshold = 0.72
                self.thresholds.critical_threshold = 0.87
                
            elif language == "ko":  # Korean: similar to Japanese
                self.max_memory_mb = int(self.max_memory_mb * 1.2)
                self.thresholds.warning_threshold = 0.72
                self.thresholds.critical_threshold = 0.87
                
            elif language == "ru":  # Russian: standard optimization
                self.max_memory_mb = int(self.max_memory_mb * 1.1)
                self.thresholds.warning_threshold = 0.7
                self.thresholds.critical_threshold = 0.85
                
            else:  # Default for English and others
                self.max_memory_mb = 1024
                self.thresholds.warning_threshold = 0.7
                self.thresholds.critical_threshold = 0.85
            
            logger.info(f"Memory optimized for language '{language}': max_memory={self.max_memory_mb}MB")
            
            return {
                "language": language,
                "max_memory_mb": self.max_memory_mb,
                "warning_threshold": self.thresholds.warning_threshold,
                "critical_threshold": self.thresholds.critical_threshold
            }
            
        except Exception as e:
            logger.error(f"Error optimizing memory for language '{language}': {e}")
            return {"error": str(e)}
    
    async def _monitor_memory(self):
        """Background memory monitoring task."""
        while True:
            try:
                await self.check_memory_usage()
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in memory monitoring: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    def _update_memory_history(self, memory_stats: MemoryStats):
        """Update memory history."""
        self.memory_history.append(memory_stats)
        
        # Keep history size manageable
        if len(self.memory_history) > self.max_history_size:
            self.memory_history = self.memory_history[-self.max_history_size:]
    
    def _get_current_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss / 1024 / 1024
        except Exception as e:
            logger.error(f"Error getting current memory: {e}")
            return 0.0
    
    def _get_threshold_status(self, memory_ratio: float) -> str:
        """Get threshold status based on memory usage."""
        if memory_ratio > self.thresholds.critical_threshold:
            return "critical"
        elif memory_ratio > self.thresholds.warning_threshold:
            return "warning"
        else:
            return "normal"
    
    async def _clear_memory_cache(self):
        """Clear memory cache."""
        try:
            # This would integrate with the actual caching service
            # For now, just log the action
            logger.debug("Clearing memory cache")
            
        except Exception as e:
            logger.error(f"Error clearing memory cache: {e}")
    
    async def _clear_translation_memory(self):
        """Clear translation memory."""
        try:
            # This would integrate with the actual translation service
            # For now, just log the action
            logger.debug("Clearing translation memory")
            
        except Exception as e:
            logger.error(f"Error clearing translation memory: {e}")
    
    async def _clear_tracked_objects(self):
        """Clear tracked objects."""
        try:
            # Clear tracked objects that are no longer needed
            objects_to_remove = []
            
            for name, obj in self.tracked_objects.items():
                # Add logic to determine if object should be cleared
                # For now, clear all tracked objects
                objects_to_remove.append(name)
            
            for name in objects_to_remove:
                self.untrack_object(name)
            
            logger.debug(f"Cleared {len(objects_to_remove)} tracked objects")
            
        except Exception as e:
            logger.error(f"Error clearing tracked objects: {e}")
    
    async def _emergency_measures(self):
        """Additional emergency memory measures."""
        try:
            # Force multiple garbage collection cycles
            for _ in range(3):
                gc.collect()
                await asyncio.sleep(0.1)
            
            # Clear all tracked objects
            self.tracked_objects.clear()
            self.object_sizes.clear()
            
            # Clear memory history
            self.memory_history.clear()
            
            logger.warning("Emergency memory measures completed")
            
        except Exception as e:
            logger.error(f"Error in emergency measures: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get memory manager status."""
        try:
            current_memory = self._get_current_memory_mb()
            memory_ratio = current_memory / self.max_memory_mb
            
            return {
                "current_memory_mb": current_memory,
                "max_memory_mb": self.max_memory_mb,
                "memory_usage_ratio": memory_ratio,
                "threshold_status": self._get_threshold_status(memory_ratio),
                "tracked_objects_count": len(self.tracked_objects),
                "cleanups_performed": self.stats["cleanups_performed"],
                "total_memory_freed_mb": self.stats["total_memory_freed_mb"],
                "memory_warnings": self.stats["memory_warnings"],
                "critical_memory_events": self.stats["critical_memory_events"],
                "auto_cleanup_enabled": self.enable_auto_cleanup,
                "last_cleanup_time": self.last_cleanup_time
            }
        except Exception as e:
            logger.error(f"Error getting memory manager status: {e}")
            return {"error": str(e)}


# Global memory manager instance
_global_memory_manager: Optional[MemoryManager] = None


def get_global_memory_manager() -> MemoryManager:
    """Get global memory manager instance."""
    global _global_memory_manager
    if _global_memory_manager is None:
        _global_memory_manager = MemoryManager()
    return _global_memory_manager


def set_global_memory_manager(manager: MemoryManager):
    """Set global memory manager instance."""
    global _global_memory_manager
    _global_memory_manager = manager
