"""
Caching Service for unified caching across all agents.
Provides in-memory and persistent caching with TTL support.
"""

import asyncio
import hashlib
import pickle
import time
from typing import Any, Dict, Optional
from pathlib import Path

import logging

# Configure logger
logger = logging.getLogger(__name__)


class CacheEntry:
    """Represents a cached item with metadata."""

    def __init__(self, value: Any, ttl: Optional[int] = None):
        self.value = value
        self.created_at = time.time()
        self.ttl = ttl
        self.access_count = 0
        self.last_accessed = self.created_at

    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl

    def access(self):
        """Mark the entry as accessed."""
        self.access_count += 1
        self.last_accessed = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'value': self.value,
            'created_at': self.created_at,
            'ttl': self.ttl,
            'access_count': self.access_count,
            'last_accessed': self.last_accessed
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheEntry':
        """Create from dictionary."""
        entry = cls(data['value'], data['ttl'])
        entry.created_at = data['created_at']
        entry.access_count = data['access_count']
        entry.last_accessed = data['last_accessed']
        return entry


class CachingService:
    """Unified caching service for all agents."""

    def __init__(self, cache_dir: Optional[str] = None, max_memory_size: int = 1000):
        self.cache_dir = Path(cache_dir or "./cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.max_memory_size = max_memory_size
        self.memory_cache: Dict[str, CacheEntry] = {}
        self.logger = logger

        # Cache statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'expired': 0
        }

    def _generate_key(self, *args, **kwargs) -> str:
        """Generate a cache key from arguments."""
        # Create a string representation of the arguments
        key_parts = []

        for arg in args:
            if isinstance(arg, (str, int, float, bool)):
                key_parts.append(str(arg))
            elif isinstance(arg, (list, tuple)):
                key_parts.append(str(sorted(arg)))
            elif isinstance(arg, dict):
                key_parts.append(str(sorted(arg.items())))
            else:
                key_parts.append(str(hash(str(arg))))

        for key, value in sorted(kwargs.items()):
            key_parts.append(f"{key}:{value}")

        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _get_cache_file_path(self, key: str) -> Path:
        """Get the file path for a cache key."""
        return self.cache_dir / f"{key}.cache"

    async def get(self, key: str, default: Any = None) -> Any:
        """Get a value from cache."""
        # Try memory cache first
        if key in self.memory_cache:
            entry = self.memory_cache[key]
            if not entry.is_expired():
                entry.access()
                self.stats['hits'] += 1
                return entry.value
            else:
                # Remove expired entry
                del self.memory_cache[key]
                self.stats['expired'] += 1

        # Try persistent cache
        cache_file = self._get_cache_file_path(key)
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    entry_data = pickle.load(f)
                    entry = CacheEntry.from_dict(entry_data)

                    if not entry.is_expired():
                        entry.access()
                        # Move to memory cache
                        self._add_to_memory_cache(key, entry)
                        self.stats['hits'] += 1
                        return entry.value
                    else:
                        # Remove expired file
                        cache_file.unlink()
                        self.stats['expired'] += 1
            except Exception as e:
                self.logger.warning(f"Error reading cache file {cache_file}: {e}")
                if cache_file.exists():
                    cache_file.unlink()

        self.stats['misses'] += 1
        return default

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a value in cache."""
        try:
            entry = CacheEntry(value, ttl)

            # Add to memory cache
            self._add_to_memory_cache(key, entry)

            # Save to persistent cache
            cache_file = self._get_cache_file_path(key)
            with open(cache_file, 'wb') as f:
                pickle.dump(entry.to_dict(), f)

            self.stats['sets'] += 1
            return True

        except Exception as e:
            self.logger.error(f"Error setting cache key {key}: {e}")
            return False

    def _add_to_memory_cache(self, key: str, entry: CacheEntry):
        """Add entry to memory cache with size management."""
        # Remove expired entries first
        expired_keys = [k for k, v in self.memory_cache.items() if v.is_expired()]
        for k in expired_keys:
            del self.memory_cache[k]
            self.stats['expired'] += 1

        # If still at capacity, remove least recently used
        if len(self.memory_cache) >= self.max_memory_size:
            lru_key = min(
                self.memory_cache.keys(),
                key=lambda k: self.memory_cache[k].last_accessed
            )
            del self.memory_cache[lru_key]

        self.memory_cache[key] = entry

    async def delete(self, key: str) -> bool:
        """Delete a value from cache."""
        try:
            # Remove from memory cache
            if key in self.memory_cache:
                del self.memory_cache[key]

            # Remove from persistent cache
            cache_file = self._get_cache_file_path(key)
            if cache_file.exists():
                cache_file.unlink()

            self.stats['deletes'] += 1
            return True

        except Exception as e:
            self.logger.error(f"Error deleting cache key {key}: {e}")
            return False

    async def clear(self, pattern: Optional[str] = None) -> int:
        """Clear cache entries, optionally matching a pattern."""
        deleted_count = 0

        # Clear memory cache
        if pattern:
            keys_to_delete = [k for k in self.memory_cache.keys() if pattern in k]
        else:
            keys_to_delete = list(self.memory_cache.keys())

        for key in keys_to_delete:
            del self.memory_cache[key]
            deleted_count += 1

        # Clear persistent cache
        if pattern:
            cache_files = [f for f in self.cache_dir.glob("*.cache") if pattern in f.stem]
        else:
            cache_files = list(self.cache_dir.glob("*.cache"))

        for cache_file in cache_files:
            try:
                cache_file.unlink()
                deleted_count += 1
            except Exception as e:
                self.logger.warning(f"Error deleting cache file {cache_file}: {e}")

        return deleted_count

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = (self.stats['hits'] / total_requests * 100) if total_requests > 0 else 0

        return {
            **self.stats,
            'hit_rate': hit_rate,
            'memory_cache_size': len(self.memory_cache),
            'persistent_cache_files': len(list(self.cache_dir.glob("*.cache"))),
            'cache_dir': str(self.cache_dir)
        }

    async def cleanup_expired(self) -> int:
        """Clean up expired cache entries."""
        cleaned_count = 0

        # Clean memory cache
        expired_keys = [k for k, v in self.memory_cache.items() if v.is_expired()]
        for key in expired_keys:
            del self.memory_cache[key]
            cleaned_count += 1

        # Clean persistent cache
        for cache_file in self.cache_dir.glob("*.cache"):
            try:
                with open(cache_file, 'rb') as f:
                    entry_data = pickle.load(f)
                    entry = CacheEntry.from_dict(entry_data)

                    if entry.is_expired():
                        cache_file.unlink()
                        cleaned_count += 1
            except Exception as e:
                self.logger.warning(f"Error reading cache file {cache_file}: {e}")
                # Remove corrupted files
                cache_file.unlink()
                cleaned_count += 1

        return cleaned_count

    def get_cache_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate a cache key with a prefix."""
        base_key = self._generate_key(*args, **kwargs)
        return f"{prefix}:{base_key}"

    async def get_or_set(self, key: str, default_func, ttl: Optional[int] = None) -> Any:
        """Get from cache or set using default function."""
        value = await self.get(key)
        if value is not None:
            return value

        # Generate default value
        if asyncio.iscoroutinefunction(default_func):
            value = await default_func()
        else:
            value = default_func()

        # Cache the value
        await self.set(key, value, ttl)
        return value


# Global instance
caching_service = CachingService()
