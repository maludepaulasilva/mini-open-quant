import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import sys

RAIZ = Path.cwd().parent if Path.cwd().name == "tests" else Path.cwd()
RAW = RAIZ / "data" / "raw"
DATA = RAIZ / "data"
sys.path.insert(0, str(RAIZ / "src"))

from moq import data

def test_validate_rejects_unsorted_index():
    idx = pd.bdate_range("2020-01-01", periods=10)
    df = pd.DataFrame({"A": 1.0}, index=idx)
    with pytest.raises(ValueError, match="fora de ordem"):
        data.validate_prices(df.iloc[::-1])


def test_clean_drops_and_ffills():
    idx = pd.bdate_range("2020-01-01", periods=100)
    df = pd.DataFrame({"bom": 10.0, "ruim": 10.0, "furo": 10.0}, index=idx)
    df.loc[idx[:10], "ruim"] = np.nan
    df.iloc[50:55, 2] = np.nan
    limpo = data.clean_prices(df)
    assert "ruim" not in limpo.columns
    assert limpo["furo"].iloc[50:53].notna().all()
    assert limpo["furo"].iloc[53:55].isna().all()

def test_align_cdi_rejects_percent():
    idx = pd.bdate_range("2020-01-01", periods=10)
    prices = pd.DataFrame({"A": 1.0}, index=idx)
    with pytest.raises(ValueError, match="divida por 100"):
        data.align_cdi(pd.Series(0.04, index=idx), prices)