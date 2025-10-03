"""
Bounded Results Cache

LRU (Least Recently Used) cache implementation for analysis results
to prevent unbounded memory growth.
"""

from collections import OrderedDict
from typing import Any, Dict, Optional
import sys
from datetime import datetime

from logging_config import get_logger

logger = get_logger(__name__)


class BoundedResultsCache:
    """
    LRU cache for analysis results with size and memory limits.
    
    Features:
    - Automatic eviction of least recently used items
    - Memory size tracking and limits
    - Figure size estimation for Plotly objects
    - Statistics tracking (hits, misses, evictions)
    
    Parameters
    ----------
    max_items : int
        Maximum number of cached items (default: 50)
    max_memory_mb : float
        Maximum memory usage in MB (default: 500 MB)
    """
    
    def __init__(self, max_items: int = 50, max_memory_mb: float = 500.0):
        """Initialize bounded cache."""
        self.max_items = max_items
        self.max_memory_bytes = int(max_memory_mb * 1024 * 1024)
        self.cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.memory_usage: Dict[str, int] = {}
        self.total_memory = 0
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'sets': 0,
            'total_items_cached': 0
        }
        
        logger.info(f"BoundedResultsCache initialized: max_items={max_items}, "
                   f"max_memory={max_memory_mb}MB")
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get item from cache (moves to end for LRU).
        
        Parameters
        ----------
        key : str
            Cache key (typically analysis type)
        
        Returns
        -------
        dict or None
            Cached results if found, None otherwise
        """
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.stats['hits'] += 1
            logger.debug(f"Cache hit: {key}")
            return self.cache[key]
        
        self.stats['misses'] += 1
        logger.debug(f"Cache miss: {key}")
        return None
    
    def set(self, key: str, value: Dict[str, Any]):
        """
        Set item in cache with LRU eviction.
        
        Parameters
        ----------
        key : str
            Cache key (typically analysis type)
        value : dict
            Analysis results to cache
        """
        # Estimate memory size of value
        value_size = self._estimate_size(value)
        
        # Check if single item exceeds max memory
        if value_size > self.max_memory_bytes:
            logger.warning(
                f"Item '{key}' size ({value_size / 1024 / 1024:.1f}MB) exceeds "
                f"max cache memory ({self.max_memory_bytes / 1024 / 1024:.1f}MB). "
                "Storing without large figures."
            )
            # Store without figures to save memory
            value = self._strip_large_figures(value, max_figure_size=10 * 1024 * 1024)
            value_size = self._estimate_size(value)
        
        # Remove existing key if present
        if key in self.cache:
            old_size = self.memory_usage.get(key, 0)
            self.total_memory -= old_size
            del self.cache[key]
            del self.memory_usage[key]
        
        # Evict items if necessary (LRU)
        while (len(self.cache) >= self.max_items or 
               self.total_memory + value_size > self.max_memory_bytes):
            if not self.cache:
                break
            
            # Remove least recently used (first item)
            evicted_key, evicted_value = self.cache.popitem(last=False)
            evicted_size = self.memory_usage.pop(evicted_key, 0)
            self.total_memory -= evicted_size
            self.stats['evictions'] += 1
            
            logger.info(f"Cache eviction: {evicted_key} "
                       f"(freed {evicted_size / 1024 / 1024:.1f}MB)")
        
        # Add new item
        self.cache[key] = value
        self.memory_usage[key] = value_size
        self.total_memory += value_size
        self.stats['sets'] += 1
        self.stats['total_items_cached'] += 1
        
        logger.debug(f"Cache set: {key} (size={value_size / 1024 / 1024:.1f}MB, "
                    f"total={self.total_memory / 1024 / 1024:.1f}MB)")
    
    def clear(self):
        """Clear entire cache."""
        items_cleared = len(self.cache)
        memory_freed = self.total_memory
        
        self.cache.clear()
        self.memory_usage.clear()
        self.total_memory = 0
        
        logger.info(f"Cache cleared: {items_cleared} items, "
                   f"{memory_freed / 1024 / 1024:.1f}MB freed")
    
    def remove(self, key: str) -> bool:
        """
        Remove specific item from cache.
        
        Parameters
        ----------
        key : str
            Cache key to remove
        
        Returns
        -------
        bool
            True if item was removed, False if not found
        """
        if key in self.cache:
            size = self.memory_usage.pop(key, 0)
            self.total_memory -= size
            del self.cache[key]
            logger.debug(f"Cache remove: {key} (freed {size / 1024 / 1024:.1f}MB)")
            return True
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns
        -------
        dict
            Statistics including hits, misses, size, memory usage
        """
        hit_rate = (self.stats['hits'] / (self.stats['hits'] + self.stats['misses']) * 100
                   if (self.stats['hits'] + self.stats['misses']) > 0 else 0)
        
        return {
            'size': len(self.cache),
            'max_items': self.max_items,
            'memory_mb': self.total_memory / 1024 / 1024,
            'max_memory_mb': self.max_memory_bytes / 1024 / 1024,
            'memory_percent': (self.total_memory / self.max_memory_bytes * 100
                              if self.max_memory_bytes > 0 else 0),
            'hits': self.stats['hits'],
            'misses': self.stats['misses'],
            'hit_rate': hit_rate,
            'evictions': self.stats['evictions'],
            'sets': self.stats['sets'],
            'total_cached': self.stats['total_items_cached'],
            'keys': list(self.cache.keys())
        }
    
    def _estimate_size(self, obj: Any, seen: Optional[set] = None) -> int:
        """
        Recursively estimate memory size of object.
        
        Parameters
        ----------
        obj : Any
            Object to measure
        seen : set, optional
            Set of already-seen object IDs to avoid circular references
        
        Returns
        -------
        int
            Estimated size in bytes
        """
        if seen is None:
            seen = set()
        
        obj_id = id(obj)
        if obj_id in seen:
            return 0
        
        seen.add(obj_id)
        size = sys.getsizeof(obj)
        
        # Handle different types
        if isinstance(obj, dict):
            size += sum(self._estimate_size(k, seen) + self._estimate_size(v, seen)
                       for k, v in obj.items())
        elif isinstance(obj, (list, tuple, set)):
            size += sum(self._estimate_size(item, seen) for item in obj)
        elif hasattr(obj, '__dict__'):
            size += self._estimate_size(obj.__dict__, seen)
        
        # Special handling for Plotly figures (can be very large)
        if hasattr(obj, '_data_objs') or hasattr(obj, 'data'):
            # Plotly figure detected - these can be 10-100MB
            # Add penalty for large figures
            size *= 10
        
        return size
    
    def _strip_large_figures(self, value: Dict[str, Any], 
                            max_figure_size: int = 10 * 1024 * 1024) -> Dict[str, Any]:
        """
        Remove large figure objects from results to save memory.
        
        Parameters
        ----------
        value : dict
            Analysis results dictionary
        max_figure_size : int
            Maximum allowed figure size in bytes
        
        Returns
        -------
        dict
            Results with large figures replaced by placeholders
        """
        import copy
        result = copy.copy(value)
        
        # Check 'figures' key
        if 'figures' in result and isinstance(result['figures'], dict):
            figures = result['figures']
            stripped_figures = {}
            
            for fig_name, fig_obj in figures.items():
                fig_size = self._estimate_size(fig_obj)
                if fig_size > max_figure_size:
                    # Replace with placeholder
                    stripped_figures[fig_name] = {
                        '__stripped__': True,
                        'reason': f'Figure too large ({fig_size / 1024 / 1024:.1f}MB)',
                        'stripped_at': datetime.now().isoformat()
                    }
                    logger.warning(f"Stripped large figure '{fig_name}' "
                                 f"({fig_size / 1024 / 1024:.1f}MB)")
                else:
                    stripped_figures[fig_name] = fig_obj
            
            result['figures'] = stripped_figures
        
        return result
    
    def __len__(self) -> int:
        """Return number of cached items."""
        return len(self.cache)
    
    def __contains__(self, key: str) -> bool:
        """Check if key is in cache."""
        return key in self.cache
    
    def __repr__(self) -> str:
        """String representation of cache."""
        return (f"BoundedResultsCache(size={len(self.cache)}/{self.max_items}, "
                f"memory={self.total_memory / 1024 / 1024:.1f}MB/"
                f"{self.max_memory_bytes / 1024 / 1024:.1f}MB)")


# Global cache instance
_global_cache: Optional[BoundedResultsCache] = None


def get_results_cache(max_items: int = 50, max_memory_mb: float = 500.0) -> BoundedResultsCache:
    """
    Get or create global results cache instance.
    
    Parameters
    ----------
    max_items : int
        Maximum cached items (only used on first call)
    max_memory_mb : float
        Maximum memory in MB (only used on first call)
    
    Returns
    -------
    BoundedResultsCache
        Global cache instance
    """
    global _global_cache
    if _global_cache is None:
        _global_cache = BoundedResultsCache(max_items, max_memory_mb)
    return _global_cache


if __name__ == "__main__":
    # Test the cache
    import numpy as np
    
    cache = BoundedResultsCache(max_items=3, max_memory_mb=10)
    
    # Add some test data
    for i in range(5):
        cache.set(f"analysis_{i}", {
            'data': np.random.randn(100, 100),
            'summary': {'mean': i, 'std': i * 0.1},
            'success': True
        })
    
    # Check stats
    stats = cache.get_stats()
    print(f"\nCache Stats:")
    print(f"  Size: {stats['size']}/{stats['max_items']}")
    print(f"  Memory: {stats['memory_mb']:.1f}/{stats['max_memory_mb']:.1f} MB")
    print(f"  Hit rate: {stats['hit_rate']:.1f}%")
    print(f"  Evictions: {stats['evictions']}")
    print(f"  Keys: {stats['keys']}")
    
    # Test retrieval (should increase hit count)
    result = cache.get('analysis_4')
    print(f"\nRetrieved: {result is not None}")
    
    # Test miss (evicted item)
    result = cache.get('analysis_0')
    print(f"Evicted item found: {result is not None}")
    
    # Final stats
    stats = cache.get_stats()
    print(f"\nFinal Hit Rate: {stats['hit_rate']:.1f}%")
