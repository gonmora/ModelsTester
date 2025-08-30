import sys
import types
import pytest
from pathlib import Path

# --- Create stub pandas module -------------------------------------------------
class DummyDataFrame(dict):
    def to_parquet(self, *args, **kwargs):
        pass


def _read_parquet(path):
    return DummyDataFrame()

pd_stub = types.ModuleType('pandas')
pd_stub.DataFrame = DummyDataFrame
pd_stub.read_parquet = _read_parquet
sys.modules.setdefault('pandas', pd_stub)

# --- Stub out external db.db_to_df module -------------------------------------
_db_pkg = types.ModuleType('db')
_db_module = types.ModuleType('db.db_to_df')


def _placeholder_db_to_df(symbol, period, start, end):
    return DummyDataFrame()

_db_module.db_to_df = _placeholder_db_to_df
_db_pkg.db_to_df = _db_module
sys.modules.setdefault('db', _db_pkg)
sys.modules.setdefault('db.db_to_df', _db_module)

# --- Stub out data_utils to avoid numpy/pandas dependencies -------------------
data_utils_stub = types.ModuleType('src.data_utils')
def _fill_stub(df, *args, **kwargs):
    return df
data_utils_stub.fill_ohlc_gaps_flat = _fill_stub
sys.modules.setdefault('src.data_utils', data_utils_stub)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import storage


def _make_df():
    return DummyDataFrame()


def test_load_dataframe_normalizes_raw_dates(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, 'DATA_DIR', tmp_path)
    called = {}

    def fake_db_to_df(symbol, period, start, end):
        called['start'] = start
        called['end'] = end
        return _make_df()

    def fake_fill(df, desde, hasta, zero_volume=True, add_flags=True):
        called['fill_start'] = desde
        called['fill_end'] = hasta
        return df

    monkeypatch.setattr(storage, 'db_to_df', fake_db_to_df)
    monkeypatch.setattr(storage, 'fill_ohlc_gaps_flat', fake_fill)
    monkeypatch.setattr(storage, 'save_dataframe', lambda name, df: None)

    storage.load_dataframe('BTC_1h_20200101_20200102')
    assert called['start'] == '2020-01-01 00:00:00'
    assert called['end'] == '2020-01-02 00:00:00'
    assert called['fill_start'] == '2020-01-01 00:00:00'
    assert called['fill_end'] == '2020-01-02 00:00:00'


def test_load_dataframe_accepts_preformatted_dates(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, 'DATA_DIR', tmp_path)
    called = {}

    def fake_db_to_df(symbol, period, start, end):
        called['start'] = start
        called['end'] = end
        return _make_df()

    def fake_fill(df, desde, hasta, zero_volume=True, add_flags=True):
        called['fill_start'] = desde
        called['fill_end'] = hasta
        return df

    monkeypatch.setattr(storage, 'db_to_df', fake_db_to_df)
    monkeypatch.setattr(storage, 'fill_ohlc_gaps_flat', fake_fill)
    monkeypatch.setattr(storage, 'save_dataframe', lambda name, df: None)

    storage.load_dataframe('BTC_1h_2020-01-01 00:00:00_2020-01-02 00:00:00')
    assert called['start'] == '2020-01-01 00:00:00'
    assert called['end'] == '2020-01-02 00:00:00'
    assert called['fill_start'] == '2020-01-01 00:00:00'
    assert called['fill_end'] == '2020-01-02 00:00:00'
