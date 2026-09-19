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


#RISK

def _serie(*trechos):
    path = np.concatenate([np.full(n, r) for n, r in trechos])
    return pd.Series(path, index=pd.bdate_range("2020-01-01", periods=len(path)))


def test_drawdown_state_machine():
    # cai 12% (→ HALF), cai a −16% (→ STOP), recupera o pico (→ FULL) sem passar por HALF na volta
    ret = _serie((50, 0.002), (60, -0.0021), (20, -0.0025), (200, 0.003))
    exp = portfolio.drawdown_control(ret, portfolio.DrawdownRule(-0.10, -0.15))
    first_half = exp[exp == 0.5].index[0]
    first_stop = exp[exp == 0.0].index[0]
    assert first_half < first_stop
    # na volta de um STOP fica-se em 0 até o novo topo: recuperação parcial não devolve HALF
    volta = exp[first_stop:]
    first_full = volta[volta == 1.0].index[0]
    assert (exp.loc[first_stop:first_full].iloc[:-1] == 0.0).all()
    assert exp.iloc[-1] == 1.0


def test_half_volta_a_valer_depois_de_novo_topo():
    # STOP → novo topo (FULL) → cai 11% de novo: o HALF reaparece, a histerese não o proíbe
    ret = _serie((50, 0.002), (80, -0.0021), (260, 0.003), (40, -0.003))
    exp = portfolio.drawdown_control(ret, portfolio.DrawdownRule(-0.10, -0.15))
    first_stop = exp[exp == 0.0].index[0]
    assert (exp[first_stop:] == 0.5).any()


def test_drawdown_control_nao_olha_o_futuro():
    # a exposição de t sai do drawdown até t−1, então mexer no último retorno não muda nada
    ret = _serie((50, 0.002), (60, -0.0021), (20, -0.0025), (200, 0.003))
    outro = ret.copy()
    outro.iloc[-1] = -0.5
    assert portfolio.drawdown_control(ret).equals(portfolio.drawdown_control(outro))


def test_rolling_risk_contribution_soma_um(rets):
    w = portfolio.risk_parity(rets)
    rc = portfolio.rolling_risk_contribution(rets, w, window=126)
    assert len(rc) == len(rets) - 126 + 1
    assert rc.sum(axis=1).sub(1.0).abs().max() < 1e-12