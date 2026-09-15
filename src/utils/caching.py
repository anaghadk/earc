import os
import pickle
import hashlib
import json
import logging
from typing import Any, Optional, Dict

logger = logging.getLogger(__name__)

class DiskCache:
    """A simple disk-based cache for potentially heavy ML operations."""
    def __init__(self, cache_dir: str, prefix: str = "cache"):
        self.cache_dir = cache_dir
        self.prefix = prefix
        if cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

    def cache_key(self, model_id: str, params_dict: Dict[str, Any], input_hash: str) -> str:
        """Generate deterministic cache key based on model, parameters, and input hash."""
        key_dict = {
            "model_id": model_id,
            "params": params_dict,
            "input": input_hash
        }
        key_str = json.dumps(key_dict, sort_keys=True)
        return hashlib.sha256(key_str.encode('utf-8')).hexdigest()

    def _get_path(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{self.prefix}_{key}.pkl")

    def get(self, key: str) -> Optional[Any]:
        """Retrieve value from cache."""
        path = self._get_path(key)
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.error(f"Failed to read cache key {key}: {e}")
                return None
        return None

    def set(self, key: str, value: Any) -> None:
        """Store value in cache."""
        path = self._get_path(key)
        try:
            with open(path, 'wb') as f:
                pickle.dump(value, f)
        except Exception as e:
            logger.error(f"Failed to write cache key {key}: {e}")

    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        return os.path.exists(self._get_path(key))

    def invalidate(self, key: str) -> None:
        """Remove key from cache."""
        path = self._get_path(key)
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception as e:
                logger.error(f"Failed to invalidate cache key {key}: {e}")

    def clear(self) -> None:
        """Clear all cache files matching prefix."""
        try:
            for f in os.listdir(self.cache_dir):
                if f.startswith(self.prefix) and f.endswith(".pkl"):
                    os.remove(os.path.join(self.cache_dir, f))
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
