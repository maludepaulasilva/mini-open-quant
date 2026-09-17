import numpy as np
import pandas as pd
import pytest
from moq import metrics


def test_annualized_return_compound():
    idx = pd.bdate_range("2020-01-01", periods=300)
    r = 0.0004
    assert metrics.annualized_return(pd.Series(r, index=idx)) == pytest.approx((1 + r) ** 252 - 1)


def test_sharpe_zero_vol_is_nan():
    idx = pd.bdate_range("2020-01-01", periods=100)
    cdi = pd.Series(0.0004, index=idx)
    assert np.isnan(metrics.sharpe(cdi.copy(), cdi))   # réplica do CDI: excesso zero


def test_drawdown_properties():
    idx = pd.bdate_range("2020-01-01", periods=200)
    r = pd.Series(np.random.default_rng(0).normal(0, 0.02, 200), index=idx)
    dd = metrics.drawdown(r)
    assert dd.iloc[0] == 0 and (dd <= 0).all() and metrics.max_drawdown(r) == dd.min()