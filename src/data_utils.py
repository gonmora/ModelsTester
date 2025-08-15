# -*- coding: utf-8 -*-
"""
Utility functions for data loading and gap filling.

This module provides functions to handle time-series OHLCV data,
including filling missing candle intervals with flat candles.
"""

from __future__ import annotations

import pandas as pd
import numpy as np


def clamp_hasta_utc(hasta: str) -> pd.Timestamp:
    """Clamp the end timestamp to no later than the current UTC time."""
    now = pd.Timestamp.utcnow().floor('min')
    h = pd.Timestamp(hasta, tz='UTC')
    return min(h, now)


def make_5m_index(t0_utc: str, t1_utc: str) -> pd.DatetimeIndex:
    """Create a 5-minute datetime index between two UTC timestamps."""
    t0 = pd.Timestamp(t0_utc, tz='UTC').ceil('5min')
    t1 = clamp_hasta_utc(t1_utc).floor('5min')
    return pd.date_range(t0, t1, freq='5min')


def fill_ohlc_gaps_flat(df: pd.DataFrame,
                        desde: str,
                        hasta: str,
                        zero_volume: bool = True,
                        add_flags: bool = True) -> pd.DataFrame:
    """
    Fill missing OHLCV data using a "flat candle" approach.

    Any missing 5-minute intervals between `desde` and `hasta` are filled with
    candles where open=high=low=close equals the previous close. Volume-related
    columns are set to zero (or NaN) if zero_volume is True. Optional flags
    marking imputed rows are added when add_flags is True.

    Args:
        df: DataFrame indexed by datetime (must be tz-aware). Must include
            columns 'open', 'high', 'low', 'close'. Volume columns are optional.
        desde: Start timestamp (string) in ISO format.
        hasta: End timestamp (string) in ISO format.
        zero_volume: If True, volume columns are set to zero for imputed rows.
        add_flags: If True, columns 'is_gap', 'is_imputed_price',
                   'is_imputed_volume', and 'gap_len' are added.

    Returns:
        A DataFrame reindexed to all 5-minute intervals with missing rows filled.
    """
    # Localize index to UTC if not already tz-aware
    if df.index.tz is None:
        df = df.tz_localize('UTC')
    # Sort and de-duplicate index
    df = df[~df.index.duplicated(keep='first')].sort_index()
    # Build the full index
    full_idx = make_5m_index(desde, hasta)
    df = df.reindex(full_idx)

    if add_flags:
        df['is_gap'] = df['open'].isna()

    # Forward fill previous close to use for flat candles
    prev_close = df['close'].ffill()
    # Identify rows with any price columns missing
    missing_price = df[['open', 'high', 'low', 'close']].isna().any(axis=1)
    # Assign flat candles where price is missing
    if missing_price.any():
        fill_vals = np.column_stack([prev_close[missing_price]] * 4)
        df.loc[missing_price, ['open', 'high', 'low', 'close']] = fill_vals

    # Handle volume-like columns
    vol_cols = [c for c in ['volume', 'quote_volume', 'n_trades',
                            'taker_buy_base_vol', 'taker_buy_quote_vol']
                if c in df.columns]
    if zero_volume and vol_cols:
        df.loc[missing_price, vol_cols] = 0.0

    if add_flags:
        df['is_imputed_price'] = missing_price
        if vol_cols:
            was_imputed_vol = pd.Series(False, index=df.index)
            was_imputed_vol.loc[missing_price] = True
            df['is_imputed_volume'] = was_imputed_vol
        # Compute continuous gap lengths
        na_run = df['is_gap'].astype(int)
        gap_len = (na_run.groupby((na_run != na_run.shift()).cumsum())
                         .cumsum()).where(df['is_gap'], 0)
        df['gap_len'] = gap_len

    return df
