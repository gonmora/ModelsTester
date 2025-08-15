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

# Ensure db.db_to_df is importable; add Framework directory to sys.path
framework_path = os.path.abspath(os.path.join(os.getcwd(), '..', 'Framework'))
if framework_path not in sys.path:
    sys.path.append(framework_path)

# Import database loader and gap filling functions
from db.db_to_df import db_to_df
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


def save_dataframe(name: str, df: pd.DataFrame) -> None:
    """Save a pandas DataFrame to disk under the given name."""
    path = _df_path(name)
    _ensure_dir_exists(path)
    df.to_parquet(path)


def load_dataframe(name: str) -> pd.DataFrame:
    """Load a pandas DataFrame with the given name.

    If the dataset is not found on disk, and the name follows the pattern
    <symbol>_<period>_<start>_<end>, fetch the data from the database via
    `db_to_df`, fill gaps with `fill_ohlc_gaps_flat`, cache it, and return.
    """
    path = _df_path(name)
    # If cached file exists, load and return
    if os.path.exists(path):
        return pd.read_parquet(path)
    # Attempt to parse dataset name
    try:
        symbol, period, start, end = name.split("_", 3)
    except ValueError:
        raise FileNotFoundError(f"DataFrame '{name}' not found at {path}")
    # Fetch raw data from the database
    df_raw = db_to_df(symbol, period, start, end)
    # Fill OHLCV gaps
    df_filled = fill_ohlc_gaps_flat(df_raw, start, end, zero_volume=True, add_flags=True)
    # Cache and return
    save_dataframe(name, df_filled)
    return df_filled


def dataframe_exists(name: str) -> bool:
    """Return True if a base dataset with the given name exists on disk."""
    return os.path.exists(_df_path(name))


# Feature helpers

def _feature_path(df_name: str, feature_name: str) -> str:
    """Return the file path for a computed feature."""
    return os.path.join(DATA_DIR, f"{df_name}__feature__{feature_name}.parquet")


def save_feature(df_name: str, feature_name: str, data: pd.DataFrame | pd.Series) -> None:
    """Save a feature DataFrame or Series to disk."""
    path = _feature_path(df_name, feature_name)
    _ensure_dir_exists(path)
    data.to_parquet(path)


def load_feature(df_name: str, feature_name: str) -> pd.DataFrame | pd.Series:
    """Load a computed feature from disk."""
    path = _feature_path(df_name, feature_name)
    if not os.path.exists(path):
                raise FileNotFoundError(f"Feature '{feature_name}' for dataset '{df_name}' not found at {path}")
    return pd.read_parquet(path)

def feature_exists(df_name: str, feature_name: str) -> bool:
    """Return True if a computed feature exists on disk."""
    return os.path.exists(_feature_path(df_name, feature_name))

# Target helpers
def _target_path(df_name: str, target_name: str) -> str:
    """Return the file path for a computed target."""
    return os.path.join(DATA_DIR, f"{df_name}__target__{target_name}.parquet")

def save_target(df_name: str, target_name: str, data: pd.Series) -> None:
    """Save a target Series to disk."""
    path = _target_path(df_name, target_name)
    _ensure_dir_exists(path)
    data.to_parquet(path)

def load_target(df_name: str, target_name: str) -> pd.Series:
    """Load a computed target from disk."""
    path = _target_path(df_name, target_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Target '{target_name}' for dataset '{df_name}' not found at {path}")
    return pd.read_parquet(path)

def target_exists(df_name: str, target_name: str) -> bool:
    """Return True if a computed target exists on disk."""
    return os.path.exists(_target_path(df_name, target_name))

    
