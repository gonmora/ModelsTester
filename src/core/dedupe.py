# -*- coding: utf-8 -*-
from __future__ import annotations
import json
import hashlib
from typing import Dict, Any, List, Tuple, Optional


def freeze(obj: Any) -> Any:
    """Recursively freeze mutable structures into hashable equivalents."""
    if isinstance(obj, dict):
        return tuple(sorted((k, freeze(v)) for k, v in obj.items()))
    if isinstance(obj, (list, tuple)):
        return tuple(freeze(x) for x in obj)
    return obj


def canonical_json(d: Dict[str, Any]) -> str:
    """Return a canonical JSON string with sorted keys and no extra whitespace."""
    return json.dumps(d, sort_keys=True, separators=(",", ":"))


def hash_json(d: Dict[str, Any]) -> str:
    """Hash a dictionary by first serializing it to canonical JSON and then computing its SHA-256 hash."""
    data = canonical_json(d).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def make_entity_id(name: str, params: Dict[str, Any], code_hash: str, extra: Optional[str] = None) -> str:
    """Create a unique identifier for an entity based on its name, parameters, code hash, and optional extra info."""
    base = {
        "name": name,
        "params": freeze(params),
        "code_hash": code_hash,
        "extra": extra,
    }
    return hash_json(base)


def make_run_key(
    df_name: str,
    split_id: str,
    target_id: str,
    feature_ids: List[str],
    model_id: str,
    eval_cfg_id: str,
    seed: int,
) -> str:
    """
    Create a unique key for a model run based on the dataset, target, features, model, evaluation config, and random seed.
    Sorting feature IDs ensures that the key is insensitive to the order of features.
    """
    payload = {
        "df": {"name": df_name, "split_id": split_id},
        "target": target_id,
        "features": sorted(feature_ids),
        "model": model_id,
        "eval": eval_cfg_id,
        "seed": seed,
    }
    return hash_json(payload)


class BloomCache:
    """
    Simple wrapper around a Bloom filter for approximate membership testing, with a fallback to a Python set.
    Uses bloom_filter2 if installed; otherwise, uses a set for exact membership.
    """

    def __init__(self) -> None:
        try:
            from bloom_filter2 import BloomFilter  # type: ignore

            self._bloom = BloomFilter(max_elements=5_000_000, error_rate=0.001)
            self._set = None
        except Exception:
            self._bloom = None
            self._set = set()

    def add(self, key: str) -> None:
        """Add a key to the cache."""
        if self._bloom is not None:
            self._bloom.add(key)
        else:
            self._set.add(key)

    def __contains__(self, key: str) -> bool:
        """Check if a key is present in the cache."""
        if self._bloom is not None:
            return key in self._bloom
        return key in self._set
