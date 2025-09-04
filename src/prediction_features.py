# -*- coding: utf-8 -*-
"""
Auto-register saved prediction (OOF) features so they are selectable by the engine.

Scans the data directory for files named:
    <DF_NAME>__feature__pred_*__oof.parquet
and registers a registry feature per unique feature name. The registered
function simply loads the feature for the given df.

This allows random selection to include these features. Engine will still
prefer loading from storage when present, so the function is mostly a stub.
"""
from __future__ import annotations

import os
from typing import Set

from .registry import registry
from . import storage


def _scan_and_register() -> int:
    data_dir = storage.DATA_DIR
    try:
        files = [f for f in os.listdir(data_dir) if f.endswith('.parquet') and '__feature__' in f]
    except Exception:
        return 0
    names: Set[str] = set()
    for f in files:
        try:
            base = os.path.splitext(f)[0]
            # Split: <df>__feature__<feature>
            if '__feature__' not in base:
                continue
            feat = base.split('__feature__', 1)[1]
            if not feat.startswith('pred_') or '__oof' not in feat:
                continue
            names.add(feat)
        except Exception:
            continue

    for name in sorted(names):
        if name in registry.features:
            continue

        @registry.register_feature(name)
        def _loader(df, __name=name):  # type: ignore[misc]
            # If the engine calls this, try load from storage using df_name
            df_name = None
            try:
                df_name = getattr(df, 'attrs', {}).get('__df_name__')
            except Exception:
                df_name = None
            if not df_name:
                raise FileNotFoundError(f"Cannot determine df_name for feature '{__name}'")
            return storage.load_feature(str(df_name), __name)

    return len(names)


try:
    _scan_and_register()
except Exception:
    pass

