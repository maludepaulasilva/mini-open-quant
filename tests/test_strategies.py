import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import sys

RAIZ = Path.cwd().parent if Path.cwd().name == "tests" else Path.cwd()
RAW = RAIZ / "data" / "raw"
DATA = RAIZ / "data"
sys.path.insert(0, str(RAIZ / "src"))

from moq import strategies

@pytest.fixture(scope="module")
def prices():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2015-01-01", periods=1500)
    r = rng.normal(0.0003, 0.015, (1500, 8))
    return pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), index=idx,
                        columns=[f"A{i}" for i in range(8)])

STRATS = {
    "momentum": (strategies.momentum, {}),
    "mean_reversion" : (strategies.mean_reversion, {}),
    "pairs" : (strategies.pairs, {"a": "A0", "b": "A1"})
    #as outras entram aqui conforme forem escritas
}


@pytest.mark.parametrize("name", list(STRATS))
def test_no_lookahead(prices, name):
    fn, kw = STRATS[name]
    t = 900
    base = fn(prices, **kw)
    shocked = prices.copy()
    shocked.iloc[t:] *= np.random.default_rng(1).uniform(0.5, 1.5, shocked.iloc[t:].shape)
    alt = fn(shocked, **kw)
    pd.testing.assert_frame_equal(base.iloc[: t + 1], alt.iloc[: t + 1])


@pytest.mark.parametrize("name", list(STRATS))
def test_contract(prices, name):
    fn, kw = STRATS[name]
    w = fn(prices, **kw)
    assert w.index.equals(prices.index)
    assert list(w.columns) == list(prices.columns)
    assert not w.isna().any().any()
    assert (w.abs().sum(axis=1) <= 1 + 1e-9).all()
    assert getattr(fn, "__moq_lagged__", False)


def test_lagged_bloqueia_mesmo_dia(prices):
    # o momentum já usa só preços passados (skip > 0), então não serve pra testar o decorador:
    # aqui a estratégia usa o preço do próprio dia, e só o shift(1) do lagged impede o look-ahead
    @strategies.lagged()
    def espiao(p):
        return strategies._normalize_rows(p)  # peso proporcional ao preço: qualquer choque muda o peso

    t = 900
    shocked = prices.copy()
    shocked.iloc[t:] *= np.random.default_rng(1).uniform(0.5, 1.5, shocked.iloc[t:].shape)
    pd.testing.assert_frame_equal(espiao(prices).iloc[: t + 1], espiao(shocked).iloc[: t + 1])

def test_carry_signal_is_lagged():
    idx = pd.bdate_range("2020-01-01", periods=400)
    cdi = pd.Series(0.0004, index= idx)
    y = pd.Series(0.11, index=idx)
    y.iloc[300:] = 0.20
    r = strategies.carry_returns(y, cdi, window=60)
    assert r.iloc[300] == pytest.approx(cdi.iloc[300])