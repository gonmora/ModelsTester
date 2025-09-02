# -*- coding: utf-8 -*-
"""Utilities for loading and saving datasets, features and targets.

This module provides simple functions for reading and writing pandas
DataFrames and Series to Parquet files. It defines a naming convention
for base datasets, features and targets so that computed artefacts
can be cached and reused across runs.

The default directory for storage can be overridden by setting the
`DATA_DIR` environment variable; otherwise, it defaults to 'data'
in the current working directory.
"""
from __future__ import annotations

import os
import sys
from typing import Optional

import pandas as pd

# Ensure db.db_to_df is importable; try several plausible Framework locations
MODULE_DIR = os.path.abspath(os.path.dirname(__file__))
_fw_candidates = []
try:
    env_fw = os.environ.get("FRAMEWORK_DIR")
    if env_fw:
        _fw_candidates.append(os.path.abspath(env_fw))
except Exception:
    pass
_fw_candidates += [
    os.path.abspath(os.path.join(MODULE_DIR, '..', 'Framework')),
    os.path.abspath(os.path.join(MODULE_DIR, '..', '..', 'Framework')),
    os.path.abspath(os.path.join(os.getcwd(), '..', 'Framework')),
]
for _p in _fw_candidates:
    try:
        if os.path.isdir(_p) and _p not in sys.path:
            sys.path.append(_p)
    except Exception:
        pass

# Import database loader and gap filling functions
try:
    from db.db_to_df import db_to_df
except Exception:
    # Allow importing reporting/engine in environments without the Framework/db package
    # Only raise if a function that actually needs db_to_df is called.
    def db_to_df(*args, **kwargs):  # type: ignore
        raise ImportError(
            "db.db_to_df is not available. This import is only required when loading raw data "
            "into a dataframe by name. Monitoring/reporting can run without it."
        )

from .data_utils import fill_ohlc_gaps_flat

# Root directory where Parquet files are stored
DATA_DIR = os.environ.get("DATA_DIR", "data")


def _ensure_dir_exists(path: str) -> None:
    """Ensure that the directory for a given file path exists."""
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


# Base dataset helpers

def _df_path(name: str) -> str:
    """Return the file path for a base dataset given its name."""
    return os.path.join(DATA_DIR, f"{name}.parquet")


def _normalize_date(date_str: str) -> str:
    """Normalize `YYYYMMDD` strings to ``YYYY-MM-DD 00:00:00``.

    Parameters
    ----------
    date_str : str
        Date string that may be in compact ``YYYYMMDD`` form.

    Returns
    -------
    str
        The date in ISO format with a time component set to midnight.
    """
    if len(date_str) == 8 and date_str.isdigit():
        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]} 00:00:00"
    return date_str


def save_dataframe(name: str, df: pd.DataFrame) -> None:
    """Save a pandas DataFrame to disk under the given name."""
    path = _df_path(name)
    _ensure_dir_exists(path)
    try:
        df.attrs['__df_name__'] = name
    except Exception:
        pass
    df.to_parquet(path)


def load_dataframe(name: str) -> pd.DataFrame:
    """Load a pandas DataFrame with the given name.

    If the dataset is not found on disk, and the name follows the pattern
    <symbol>_<period>_<start>_<end>, fetch the data from the database via
    `db_to_df`, fill gaps with `fill_ohlc_gaps_flat`, cache it, and return.
    """
    # 0) Allow passing a direct parquet path
    try:
        if isinstance(name, str) and name.endswith('.parquet') and os.path.exists(name):
            df = pd.read_parquet(name)
            try:
                df.attrs['__df_name__'] = os.path.splitext(os.path.basename(name))[0]
            except Exception:
                pass
            return df
    except Exception:
        pass
    path = _df_path(name)
    # If cached file exists, load and return
    if os.path.exists(path):
        df = pd.read_parquet(path)
        try:
            df.attrs['__df_name__'] = name
        except Exception:
            pass
        return df
    # Attempt to parse dataset name
    try:
        symbol, period, start, end = name.split("_", 3)
    except ValueError:
        raise FileNotFoundError(f"DataFrame '{name}' not found at {path}")
    # Normalize compact date strings if necessary
    start = _normalize_date(start)
    end = _normalize_date(end)
    # Fetch raw data from the database
    df_raw = db_to_df(symbol, period, start, end)
    # Fill OHLCV gaps
    df_filled = fill_ohlc_gaps_flat(df_raw, start, end, zero_volume=True, add_flags=True)
    # Cache and return
    save_dataframe(name, df_filled)
    try:
        df_filled.attrs['__df_name__'] = name
    except Exception:
        pass
    return df_filled


def dataframe_exists(name: str) -> bool:
    """Return True if a base dataset with the given name exists on disk."""
    return os.path.exists(_df_path(name))


# Feature helpers

def _feature_path(df_name: str, feature_name: str) -> str:
    """Return the file path for a computed feature."""
    return os.path.join(DATA_DIR, f"{df_name}__feature__{feature_name}.parquet")


def save_feature(df_name: str, feature_name: str, data: pd.DataFrame | pd.Series) -> None:
    """Save a feature DataFrame or Series to disk.

    If ``data`` is a Series, it is stored as a single-column DataFrame with
    column name set to the feature name to ensure compatibility across pandas versions.
    """
    path = _feature_path(df_name, feature_name)
    _ensure_dir_exists(path)
    if isinstance(data, pd.Series):
        df = data.to_frame(name=feature_name)
    else:
        df = data
    df.to_parquet(path)


def load_feature(df_name: str, feature_name: str) -> pd.DataFrame | pd.Series:
    """Load a computed feature from disk.

    Returns a Series if the stored parquet has a single column; otherwise a DataFrame.
    """
    path = _feature_path(df_name, feature_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature '{feature_name}' for dataset '{df_name}' not found at {path}")
    df = pd.read_parquet(path)
    try:
        # Return Series if single-column
        if hasattr(df, 'shape') and getattr(df, 'shape', (0, 0))[1] == 1:
            return df.iloc[:, 0]
    except Exception:
        pass
    return df

def feature_exists(df_name: str, feature_name: str) -> bool:
    """Return True if a computed feature exists on disk."""
    return os.path.exists(_feature_path(df_name, feature_name))

# Target helpers
def _target_path(df_name: str, target_name: str) -> str:
    """Return the file path for a computed target."""
    return os.path.join(DATA_DIR, f"{df_name}__target__{target_name}.parquet")

def save_target(df_name: str, target_name: str, data: pd.Series) -> None:
    """Save a target Series to disk.

    Stored as a single-column DataFrame to be robust to pandas versions without Series.to_parquet.
    """
    path = _target_path(df_name, target_name)
    _ensure_dir_exists(path)
    df = data.to_frame(name=target_name)
    df.to_parquet(path)

def load_target(df_name: str, target_name: str) -> pd.Series | pd.DataFrame:
    """Load a computed target from disk.

    Returns a Series if the stored parquet has a single column; otherwise a DataFrame.
    """
    path = _target_path(df_name, target_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Target '{target_name}' for dataset '{df_name}' not found at {path}")
    df = pd.read_parquet(path)
    try:
        if hasattr(df, 'shape') and getattr(df, 'shape', (0, 0))[1] == 1:
            return df.iloc[:, 0]
    except Exception:
        pass
    return df

def target_exists(df_name: str, target_name: str) -> bool:
    """Return True if a computed target exists on disk."""
    return os.path.exists(_target_path(df_name, target_name))

    
