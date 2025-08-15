# -*- coding: utf-8 -*-
"""
Basic components (targets, features, models) for the ModelsTester project.

This module defines simple baseline target and feature functions and a basic
classification model. These are registered with the global registry on import.
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from .registry import registry

# Target: binary label indicating if the next close price is higher
@registry.register_target('future_up')
def future_up(df: pd.DataFrame) -> pd.Series:
    """Return 1 if the next closing price is higher than the current close, else 0."""
    y = (df['close'].shift(-1) > df['close']).astype(int)
    y.name = 'future_up'
    return y

# Feature 1: Percentage change of closing price
@registry.register_feature('pct_change')
def pct_change_feature(df: pd.DataFrame) -> pd.Series:
    x = df['close'].pct_change().fillna(0)
    x.name = 'pct_change'
    return x

# Feature 2: 3-period momentum (difference between current close and close 3 periods ago)
@registry.register_feature('momentum_3')
def momentum_3_feature(df: pd.DataFrame) -> pd.Series:
    x = df['close'] - df['close'].shift(3)
    x = x.fillna(0)
    x.name = 'momentum_3'
    return x

# Feature 3: 5-period rolling mean of closing price
@registry.register_feature('rolling_mean_5')
def rolling_mean_5_feature(df: pd.DataFrame) -> pd.Series:
    x = df['close'].rolling(window=5).mean().fillna(method='bfill')
    x.name = 'rolling_mean_5'
    return x

# Feature 4: 5-period rolling standard deviation of closing price
@registry.register_feature('rolling_std_5')
def rolling_std_5_feature(df: pd.DataFrame) -> pd.Series:
    x = df['close'].rolling(window=5).std().fillna(method='bfill')
    x.name = 'rolling_std_5'
    return x

# Feature 5: Percentage change in volume (or zeros if volume not available)
@registry.register_feature('volume_change')
def volume_change_feature(df: pd.DataFrame) -> pd.Series:
    if 'volume' in df.columns:
        x = df['volume'].pct_change().fillna(0)
    else:
        x = pd.Series(0.0, index=df.index)
    x.name = 'volume_change'
    return x

# Model: simple logistic regression baseline with sklearn if available, otherwise random baseline
@registry.register_model('logistic_baseline')
def logistic_baseline(
    y: pd.Series,
    features: dict[str, pd.Series],
    df: pd.DataFrame,
    selection: dict[str, any],
):
    """Train a logistic regression model to predict the binary target from features.

    Args:
        y: Target series (binary 0/1).
        features: Dictionary of feature series.
        df: Original dataframe (unused but kept for API consistency).
        selection: Selection dictionary with component names (unused here).

    Returns:
        Tuple of (model, metrics dict). If sklearn is unavailable, returns (None, metrics).
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score

        # Prepare feature matrix
        X = pd.concat(features, axis=1)
        combined = pd.concat([y, X], axis=1).dropna()
        y_clean = combined.iloc[:, 0]
        X_clean = combined.iloc[:, 1:]

        # If not enough data or only one class, return NaN metric
        if y_clean.nunique() < 2 or len(X_clean) < 10:
            return None, {'auc': float('nan')}

        split = int(len(X_clean) * 0.8)
        X_train = X_clean.iloc[:split]
        X_test = X_clean.iloc[split:]
        y_train = y_clean.iloc[:split]
        y_test = y_clean.iloc[split:]

        model = LogisticRegression(max_iter=200)
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probs)
        return model, {'auc': auc}
    except Exception:
        # Fallback: random predictions baseline
        y_nonan = y.dropna()
        if len(y_nonan) == 0:
            return None, {'accuracy': float('nan')}
        preds = np.random.choice([0, 1], size=len(y_nonan))
        acc = float((preds == y_nonan.values).mean())
        return None, {'accuracy': acc}
