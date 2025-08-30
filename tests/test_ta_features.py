import pytest

pd = pytest.importorskip("pandas")
ta = pytest.importorskip("pandas_ta")

import src.ta_features  # noqa: F401 - triggers registration on import
from src.registry import registry


def _sample_df():
    return pd.DataFrame(
        {
            "open": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "high": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            "low": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "close": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "volume": [1] * 10,
        }
    )


def test_macd_registers_each_column():
    df = _sample_df()
    expected = ta.momentum.macd(df["close"])
    for col in expected.columns:
        fname = f"ta_momentum_macd_{col}"
        assert fname in registry.features
        out = registry.get_feature(fname)(df)
        assert out.name == fname
        assert out.equals(expected[col])

