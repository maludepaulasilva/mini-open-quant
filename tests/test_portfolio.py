import numpy as np
import pandas as pd
import pytest
from moq import portfolio


@pytest.fixture
def rets():
    rng = np.random.default_rng(5)
    idx = pd.bdate_range("2018-01-01", periods=1000)
    return pd.DataFrame({"hi": rng.normal(0.0004, 0.02, 1000), "lo": rng.normal(0.0002, 0.01, 1000),
                         "mid": rng.normal(0.0003, 0.015, 1000)}, index=idx)


def test_all_allocations_on_simplex(rets):
    cdi = pd.Series(0.0003, index=rets.index)
    for w in (portfolio.equal_weight(rets), portfolio.risk_parity(rets), portfolio.max_sharpe(rets, cdi)):
        assert (w >= -1e-9).all() and w.sum() == pytest.approx(1.0, abs=1e-9)


def test_risk_parity_inverse_vol(rets):
    w = portfolio.risk_parity(rets)
    assert w["lo"] / w["hi"] == pytest.approx(rets["hi"].std() / rets["lo"].std(), rel=1e-9)


def test_max_sharpe_respects_cap(rets):
    cdi = pd.Series(0.0003, index=rets.index)
    assert (portfolio.max_sharpe(rets, cdi, w_max=0.5) <= 0.5 + 1e-9).all()
    with pytest.raises(ValueError):
        portfolio.max_sharpe(rets, cdi, w_max=0.2)     # 3 × 0.2 < 1


def test_risk_contributions_sum_to_variance(rets):
    w = portfolio.risk_parity(rets)
    rc = portfolio.risk_contribution(rets, w)
    assert rc.sum() == pytest.approx(float(w @ (rets.cov() * 252) @ w), abs=1e-12)