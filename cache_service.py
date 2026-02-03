"""
Cache Service: Simple in-memory caching for API responses.
Reduces database queries and improves response times.
"""

import hashlib
import json
import time
import threading
from typing import Any, Dict, Optional


class CacheService:
    """
    Cache en memoire thread-safe avec TTL.
    Evite les rechargements constants de la bibliotheque.
    """

    def __init__(self, default_ttl: int = 300):
        """
        Initialize cache service.

        Args:
            default_ttl: Default time-to-live in seconds (5 minutes)
        """
        self._cache: Dict[str, tuple] = {}  # {key: (data, timestamp, ttl)}
        self._default_ttl = default_ttl
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        """
        Recuperer donnees depuis le cache.

        Args:
            key: Cache key

        Returns:
            Cached data or None if expired/not found
        """
        with self._lock:
            if key not in self._cache:
                return None

            data, timestamp, ttl = self._cache[key]
            if time.time() - timestamp > ttl:
                # Expired - remove from cache
                del self._cache[key]
                return None

            return data

    def set(self, key: str, data: Any, ttl: Optional[int] = None) -> None:
        """
        Stocker donnees dans le cache.

        Args:
            key: Cache key
            data: Data to cache
            ttl: Time-to-live in seconds (uses default if not specified)
        """
        with self._lock:
            effective_ttl = ttl if ttl is not None else self._default_ttl
            self._cache[key] = (data, time.time(), effective_ttl)

    def invalidate(self, pattern: Optional[str] = None) -> int:
        """
        Invalider le cache.

        Args:
            pattern: If provided, only invalidate keys containing this pattern.
                     If None, clear entire cache.

        Returns:
            Number of keys invalidated
        """
        with self._lock:
            if pattern is None:
                count = len(self._cache)
                self._cache.clear()
                return count

            keys_to_delete = [k for k in self._cache.keys() if pattern in k]
            for k in keys_to_delete:
                del self._cache[k]
            return len(keys_to_delete)

    def invalidate_exact(self, key: str) -> bool:
        """
        Invalider une cle specifique.

        Args:
            key: Exact cache key to invalidate

        Returns:
            True if key was found and removed
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            now = time.time()
            valid_count = 0
            expired_count = 0

            for key, (data, timestamp, ttl) in self._cache.items():
                if now - timestamp <= ttl:
                    valid_count += 1
                else:
                    expired_count += 1

            return {
                "total_keys": len(self._cache),
                "valid_keys": valid_count,
                "expired_keys": expired_count,
                "default_ttl": self._default_ttl,
            }

    def cleanup_expired(self) -> int:
        """
        Remove all expired entries from cache.

        Returns:
            Number of entries removed
        """
        with self._lock:
            now = time.time()
            expired_keys = [
                key
                for key, (data, timestamp, ttl) in self._cache.items()
                if now - timestamp > ttl
            ]
            for key in expired_keys:
                del self._cache[key]
            return len(expired_keys)

    @staticmethod
    def generate_key(*args, **kwargs) -> str:
        """
        Generate a cache key from arguments.

        Args:
            *args: Positional arguments to include in key
            **kwargs: Keyword arguments to include in key

        Returns:
            MD5 hash string as cache key
        """
        key_data = {"args": args, "kwargs": kwargs}
        key_string = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_string.encode()).hexdigest()


# Global cache instance
# TTL de 5 minutes par defaut - les donnees ne changent pas souvent
cache = CacheService(default_ttl=300)


# Specific cache instances for different data types
inventory_cache = CacheService(default_ttl=300)  # 5 min for inventory
series_cache = CacheService(default_ttl=300)  # 5 min for series
tmdb_cache = CacheService(default_ttl=3600)  # 1 hour for TMDB data (rarely changes)
