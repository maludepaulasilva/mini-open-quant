import numpy as np
import pandas as pd
import pytest
from moq import backtest, strategies

def test_analytic_two_assets():
    idx = pd.bdate_range("2021-01-01", periods=3)
    prices = pd.DataFrame({"A": [100, 110, 99], "B": [50, 50, 55]}, index=idx, dtype=float)
    w = pd.DataFrame({"A": [0, 0.5, 0.5], "B": [0, 0.5, 0.5]}, index=idx)
    cdi = pd.Series(0.0, index=idx)
    res = backtest.run(w, prices, cdi, cost_bps=0)

    assert res.ret.iloc[1] == pytest.approx(0.05, abs=1e-12)
    assert res.ret.iloc[2] == pytest.approx(0.0, abs=1e-12)


def test_cost_proportional_to_turnover():
    idx = pd.bdate_range("2021-01-01", periods=10)
    prices = pd.DataFrame({"A": 100.0, "B": 100.0}, index=idx)
    cdi = pd.Series(0.0, index=idx)
    w_const = pd.DataFrame({"A": 0.5, "B": 0.5}, index=idx)
    assert (backtest.run(w_const, prices, cdi, 10).cost.iloc[1:] == 0).all()
    w_flip = w_const.copy() ; w_flip.iloc[1] = [1.0, 0.0]
    assert backtest.run(w_flip, prices, cdi, 10).cost.iloc[1] == pytest.approx(10 / 1e4)

def test_idle_cash_earns_cdi():
    idx = pd.bdate_range("2021-01-01", periods=10)
    prices = pd.DataFrame({"A": 100.0}, index=idx)
    cdi = pd.Series(0.0004, index=idx)
    res = backtest.run(pd.DataFrame({"A": 0.0}, index=idx), prices, cdi)
    pd.testing.assert_series_equal(res.ret, cdi, check_names=False)


def test_rejects_misaligned():
    idx = pd.bdate_range("2021-01-01", periods=10)
    prices = pd.DataFrame({"A": 100.0}, index=idx)
    with pytest.raises(ValueError):
        backtest.run(pd.DataFrame({"A": 0.5}, index=idx[:-1]), prices, pd.Series(0.0, index=idx))

def test_custo_diminui_o_retorno():
    """net = gross − cost + cash. Com preço parado e CDI zero, sobra só o custo — negativo."""
    idx = pd.bdate_range("2021-01-01", periods=3)
    prices = pd.DataFrame({"A": 100.0}, index=idx)        # preço parado → gross = 0
    cdi = pd.Series(0.0, index=idx)                       # sem caixa remunerado → cash = 0
    w = pd.DataFrame({"A": [0.0, 1.0, 1.0]}, index=idx)   # turnover 1 no dia 1, 0 nos outros

    res = backtest.run(w, prices, cdi, cost_bps=10)
    assert res.cost.iloc[1] == pytest.approx(10 / 1e4)
    assert res.ret.iloc[1] == pytest.approx(-10 / 1e4)    # o dia do giro tem que dar prejuízo

    # e a identidade inteira, dia a dia, com os três componentes que o resultado devolve
    pd.testing.assert_series_equal(res.ret, res.gross - res.cost + res.cash, check_names=False)


def test_grid_produto_cartesiano():
    g = backtest._grid({"a": [1, 2], "b": [3, 4, 5]})
    assert len(g) == 6
    assert {"a": 1, "b": 3} in g and {"a": 2, "b": 5} in g


def test_walk_forward_disjoint_and_oos_only():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2015-01-01", periods=1500)
    prices = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, (1500, 8)), axis=0)),
                          index=idx, columns=[f"A{i}" for i in range(8)])
    cdi = pd.Series(0.0004, index=idx)
    oos, folds = backtest.walk_forward(strategies.momentum, prices, cdi,
                                       {"lookback": [126, 252], "top": [0.2, 0.3]},
                                       train_years=2, test_years=1, warmup=260)
    assert len(folds) >= 2
    for f in folds:
        assert f.train_end < f.test_start
    assert oos.index.min() == folds[0].test_start and oos.index.is_unique


def test_walk_forward_exige_historico():
    idx = pd.bdate_range("2015-01-01", periods=260)          # um ano só
    prices = pd.DataFrame({"A": 100.0}, index=idx)
    with pytest.raises(ValueError):
        backtest.walk_forward(strategies.momentum, prices, pd.Series(0.0, index=idx),
                              {"lookback": [126]}, train_years=3, test_years=1)
