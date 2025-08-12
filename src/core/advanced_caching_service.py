"""
Advanced multi-level caching service for multilingual content processing.
Implements memory, disk, and distributed caching strategies for optimal performance.
"""

import asyncio
import json
import os
import time
import hashlib
from typing import Any, Dict, Optional, Union
from datetime import datetime, timedelta
import pickle
import sqlite3
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DiskCache:
    """Disk-based caching for persistent storage of large datasets."""
    
    def __init__(self, cache_dir: str = "cache/disk_cache", max_size_mb: int = 1024):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_mb = max_size_mb
        self.db_path = self.cache_dir / "cache.db"
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for cache metadata."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_metadata (
                    key TEXT PRIMARY KEY,
                    file_path TEXT,
                    size_bytes INTEGER,
                    created_at TIMESTAMP,
                    accessed_at TIMESTAMP,
                    language TEXT,
                    entity_type TEXT
                )
            """)
            conn.commit()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from disk cache."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT file_path FROM cache_metadata WHERE key = ?",
                    (key,)
                )
                result = cursor.fetchone()
                
                if result:
                    file_path = Path(result[0])
                    if file_path.exists():
                        # Update access time
                        conn.execute(
                            "UPDATE cache_metadata SET accessed_at = ? WHERE key = ?",
                            (datetime.now(), key)
                        )
                        conn.commit()
                        
                        # Load data from file
                        with open(file_path, 'rb') as f:
                            return pickle.load(f)
            
            return None
        except Exception as e:
            logger.error(f"Error reading from disk cache: {e}")
            return None
    
    async def set(self, key: str, value: Any, language: str = "en", entity_type: str = "general") -> bool:
        """Set value in disk cache."""
        try:
            # Generate file path
            file_hash = hashlib.md5(key.encode()).hexdigest()
            file_path = self.cache_dir / f"{file_hash}.pkl"
            
            # Save data to file
            with open(file_path, 'wb') as f:
                pickle.dump(value, f)
            
            # Get file size
            size_bytes = file_path.stat().st_size
            
            # Update database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO cache_metadata 
                    (key, file_path, size_bytes, created_at, accessed_at, language, entity_type)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (key, str(file_path), size_bytes, datetime.now(), datetime.now(), language, entity_type))
                conn.commit()
            
            # Check cache size and cleanup if needed
            await self._cleanup_if_needed()
            return True
            
        except Exception as e:
            logger.error(f"Error writing to disk cache: {e}")
            return False
    
    async def _cleanup_if_needed(self):
        """Clean up old cache entries if size limit exceeded."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get total cache size
                cursor = conn.execute("SELECT SUM(size_bytes) FROM cache_metadata")
                total_size = cursor.fetchone()[0] or 0
                
                if total_size > self.max_size_mb * 1024 * 1024:
                    # Remove oldest entries
                    cursor = conn.execute("""
                        SELECT file_path FROM cache_metadata 
                        ORDER BY accessed_at ASC 
                        LIMIT 10
                    """)
                    old_files = cursor.fetchall()
                    
                    for (file_path,) in old_files:
                        try:
                            Path(file_path).unlink()
                            conn.execute("DELETE FROM cache_metadata WHERE file_path = ?", (file_path,))
                        except Exception as e:
                            logger.warning(f"Error removing cache file {file_path}: {e}")
                    
                    conn.commit()
                    
        except Exception as e:
            logger.error(f"Error during cache cleanup: {e}")


class DistributedCache:
    """Distributed caching for multi-instance deployments."""
    
    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url
        self.redis_client = None
        self._init_redis()
    
    def _init_redis(self):
        """Initialize Redis client if available."""
        try:
            if self.redis_url:
                import redis.asyncio as redis
                self.redis_client = redis.from_url(self.redis_url)
                logger.info("Redis client initialized for distributed caching")
            else:
                logger.info("No Redis URL provided, distributed caching disabled")
        except ImportError:
            logger.warning("Redis not available, distributed caching disabled")
        except Exception as e:
            logger.error(f"Error initializing Redis: {e}")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from distributed cache."""
        if not self.redis_client:
            return None
        
        try:
            value = await self.redis_client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Error reading from distributed cache: {e}")
            return None
    
    async def set(self, key: str, value: Any, expire_seconds: int = 3600) -> bool:
        """Set value in distributed cache."""
        if not self.redis_client:
            return False
        
        try:
            await self.redis_client.setex(
                key, 
                expire_seconds, 
                json.dumps(value, default=str)
            )
            return True
        except Exception as e:
            logger.error(f"Error writing to distributed cache: {e}")
            return False


class MultiLevelCache:
    """Multi-level caching system with memory, disk, and distributed layers."""
    
    def __init__(self, 
                 max_memory_size: int = 1000,
                 disk_cache_dir: str = "cache/disk_cache",
                 max_disk_size_mb: int = 1024,
                 redis_url: Optional[str] = None):
        self.memory_cache = {}
        self.memory_timestamps = {}
        self.max_memory_size = max_memory_size
        self.disk_cache = DiskCache(disk_cache_dir, max_disk_size_mb)
        self.distributed_cache = DistributedCache(redis_url)
        self.stats = {
            "memory_hits": 0,
            "disk_hits": 0,
            "distributed_hits": 0,
            "misses": 0,
            "total_requests": 0
        }
    
    async def get(self, key: str, language: str = "en") -> Optional[Any]:
        """Get value from multi-level cache."""
        self.stats["total_requests"] += 1
        
        # Try memory cache first
        if key in self.memory_cache:
            self.stats["memory_hits"] += 1
            self.memory_timestamps[key] = time.time()
            return self.memory_cache[key]
        
        # Try disk cache
        disk_result = await self.disk_cache.get(key)
        if disk_result is not None:
            self.stats["disk_hits"] += 1
            # Add to memory cache
            await self._add_to_memory_cache(key, disk_result)
            return disk_result
        
        # Try distributed cache
        dist_result = await self.distributed_cache.get(key)
        if dist_result is not None:
            self.stats["distributed_hits"] += 1
            # Add to disk and memory cache
            await self.disk_cache.set(key, dist_result, language)
            await self._add_to_memory_cache(key, dist_result)
            return dist_result
        
        self.stats["misses"] += 1
        return None
    
    async def set(self, key: str, value: Any, language: str = "en", entity_type: str = "general") -> bool:
        """Set value in multi-level cache."""
        try:
            # Add to memory cache
            await self._add_to_memory_cache(key, value)
            
            # Add to disk cache
            await self.disk_cache.set(key, value, language, entity_type)
            
            # Add to distributed cache
            await self.distributed_cache.set(key, value)
            
            return True
        except Exception as e:
            logger.error(f"Error setting cache value: {e}")
            return False
    
    async def _add_to_memory_cache(self, key: str, value: Any):
        """Add value to memory cache with size management."""
        # Check if cache is full
        if len(self.memory_cache) >= self.max_memory_size:
            # Remove oldest entry
            oldest_key = min(self.memory_timestamps.keys(), 
                           key=lambda k: self.memory_timestamps[k])
            del self.memory_cache[oldest_key]
            del self.memory_timestamps[oldest_key]
        
        # Add new entry
        self.memory_cache[key] = value
        self.memory_timestamps[key] = time.time()
    
    async def invalidate(self, key: str) -> bool:
        """Invalidate cache entry across all levels."""
        try:
            # Remove from memory cache
            if key in self.memory_cache:
                del self.memory_cache[key]
                del self.memory_timestamps[key]
            
            # Remove from disk cache
            with sqlite3.connect(self.disk_cache.db_path) as conn:
                conn.execute("DELETE FROM cache_metadata WHERE key = ?", (key,))
                conn.commit()
            
            # Remove from distributed cache
            if self.distributed_cache.redis_client:
                await self.distributed_cache.redis_client.delete(key)
            
            return True
        except Exception as e:
            logger.error(f"Error invalidating cache: {e}")
            return False
    
    async def clear_all(self) -> bool:
        """Clear all cache levels."""
        try:
            # Clear memory cache
            self.memory_cache.clear()
            self.memory_timestamps.clear()
            
            # Clear disk cache
            with sqlite3.connect(self.disk_cache.db_path) as conn:
                conn.execute("DELETE FROM cache_metadata")
                conn.commit()
            
            # Clear distributed cache
            if self.distributed_cache.redis_client:
                await self.distributed_cache.redis_client.flushdb()
            
            return True
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.stats["total_requests"]
        if total_requests == 0:
            hit_rate = 0.0
        else:
            hits = (self.stats["memory_hits"] + self.stats["disk_hits"] + 
                   self.stats["distributed_hits"])
            hit_rate = hits / total_requests
        
        return {
            "memory_cache_size": len(self.memory_cache),
            "memory_hits": self.stats["memory_hits"],
            "disk_hits": self.stats["disk_hits"],
            "distributed_hits": self.stats["distributed_hits"],
            "misses": self.stats["misses"],
            "total_requests": total_requests,
            "hit_rate": hit_rate,
            "memory_hit_rate": (self.stats["memory_hits"] / total_requests 
                              if total_requests > 0 else 0.0),
            "disk_hit_rate": (self.stats["disk_hits"] / total_requests 
                            if total_requests > 0 else 0.0),
            "distributed_hit_rate": (self.stats["distributed_hits"] / total_requests 
                                   if total_requests > 0 else 0.0)
        }
    
    async def cleanup_old_entries(self, max_age_hours: int = 24) -> int:
        """Clean up cache entries older than specified age."""
        try:
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            cleaned_count = 0
            
            # Clean memory cache
            current_time = time.time()
            keys_to_remove = []
            for key, timestamp in self.memory_timestamps.items():
                if current_time - timestamp > max_age_hours * 3600:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.memory_cache[key]
                del self.memory_timestamps[key]
                cleaned_count += 1
            
            # Clean disk cache
            with sqlite3.connect(self.disk_cache.db_path) as conn:
                cursor = conn.execute(
                    "SELECT key, file_path FROM cache_metadata WHERE created_at < ?",
                    (cutoff_time,)
                )
                old_entries = cursor.fetchall()
                
                for key, file_path in old_entries:
                    try:
                        Path(file_path).unlink()
                        conn.execute("DELETE FROM cache_metadata WHERE key = ?", (key,))
                        cleaned_count += 1
                    except Exception as e:
                        logger.warning(f"Error removing old cache file {file_path}: {e}")
                
                conn.commit()
            
            logger.info(f"Cleaned up {cleaned_count} old cache entries")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Error during cache cleanup: {e}")
            return 0


# Global cache instance
_global_cache: Optional[MultiLevelCache] = None


def get_global_cache() -> MultiLevelCache:
    """Get global cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = MultiLevelCache()
    return _global_cache


def set_global_cache(cache: MultiLevelCache):
    """Set global cache instance."""
    global _global_cache
    _global_cache = cache
