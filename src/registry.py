# -*- coding: utf-8 -*-
"""Simple registry for targets, features, models and evaluation configurations.

This module defines a `Registry` class that allows functions and
configurations to be registered under string names. It provides
decorators to register target functions, feature functions and
model-building functions, as well as a method to store arbitrary
configuration dictionaries for evaluation.

A global instance ``registry`` is provided for convenience.
"""

from __future__ import annotations

from typing import Callable, Dict, Any, Optional


class Registry:
    """Container for named callables and configuration dictionaries."""

    def __init__(self) -> None:
        self.targets: Dict[str, Callable[..., Any]] = {}
        self.features: Dict[str, Callable[..., Any]] = {}
        self.models: Dict[str, Callable[..., Any]] = {}
        self.eval_cfgs: Dict[str, Dict[str, Any]] = {}

    # Registration decorators
    def register_target(self, name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator to register a target-generating function."""

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            if name in self.targets:
                raise ValueError(f"Target '{name}' is already registered")
            self.targets[name] = func
            return func

        return decorator

    def register_feature(self, name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator to register a feature-generating function."""

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            if name in self.features:
                raise ValueError(f"Feature '{name}' is already registered")
            self.features[name] = func
            return func

        return decorator

    def register_model(self, name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator to register a model-building function."""

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            if name in self.models:
                raise ValueError(f"Model '{name}' is already registered")
            self.models[name] = func
            return func

        return decorator

    # Evaluation config registration
    def register_eval_cfg(self, name: str, cfg: Dict[str, Any]) -> None:
        """Register an evaluation configuration dictionary under a name."""
        if name in self.eval_cfgs:
            raise ValueError(f"Eval config '{name}' is already registered")
        self.eval_cfgs[name] = cfg

    # Retrieval helpers
    def get_target(self, name: str) -> Callable[..., Any]:
        if name not in self.targets:
            raise KeyError(f"Target '{name}' is not registered")
        return self.targets[name]

    def get_feature(self, name: str) -> Callable[..., Any]:
        if name not in self.features:
            raise KeyError(f"Feature '{name}' is not registered")
        return self.features[name]

    def get_model(self, name: str) -> Callable[..., Any]:
        if name not in self.models:
            raise KeyError(f"Model '{name}' is not registered")
        return self.models[name]

    def get_eval_cfg(self, name: str) -> Dict[str, Any]:
        if name not in self.eval_cfgs:
            raise KeyError(f"Eval config '{name}' is not registered")
        return self.eval_cfgs[name]


# Create a module-level registry for convenience
registry = Registry()
