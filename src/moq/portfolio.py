"""Portfólio - ranqueamento, alocação entre estratégias, risco."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from .metrics import TRADING_DAYS


def _check_simplex(w) -> None:
    """Pesos não negativos que somam 1. Lança se não."""
    w = np.asarray(w, dtype=float)
    if (w < -1e-9).any():
        raise AssertionError("peso negativo")
    if abs(w.sum() - 1.0) > 1e-6:
        raise AssertionError(f"pesos somam {w.sum():.6f}, não 1")


def equal_weight(rets: pd.DataFrame) -> pd.Series:
    w = pd.Series(1.0 / rets.shape[1], index=rets.columns)
    _check_simplex(w)
    return w

def combine(rets: pd.DataFrame, w: pd.Series) -> pd.Series:
    """Retorno do portfólio com pesos fixos (rebalanceamento diário implícito)."""
    _check_simplex(w.reindex(rets.columns))
    return (rets * w.reindex(rets.columns)).sum(axis=1)


def risk_parity(rets: pd.DataFrame, window: int | None = None) -> pd.Series:
    """Pesos ∝ 1/vol. window=None usa todo o histórico. Vol zero ⇒ peso 0."""
    sample = rets if window is None else rets.tail(window)
    vol = sample.std(ddof=1) * np.sqrt(TRADING_DAYS)
    inv = (1.0 / vol).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    if inv.sum() <= 0:
        raise ValueError("todas as estratégias com vol nula")
    w = inv / inv.sum()
    _check_simplex(w)
    return w


def risk_contribution(rets: pd.DataFrame, w: pd.Series) -> pd.Series:
    """RC_i = w_i·(Σw)_i.  Propriedade: Σ RC_i = variância do portfólio."""
    cov = rets.cov() * TRADING_DAYS
    wv = w.reindex(rets.columns).to_numpy()
    rc = wv * (cov.to_numpy() @ wv)
    out = pd.Series(rc, index=rets.columns)
    assert abs(out.sum() - float(wv @ cov.to_numpy() @ wv)) < 1e-12
    return out


def max_sharpe(rets: pd.DataFrame, cdi: pd.Series, w_max: float = 0.35, shrink: float = 0.5) -> pd.Series:
    """Máximo Sharpe com teto de peso e encolhimento das médias para a média comum."""
    if not (0 < w_max <= 1) or w_max * rets.shape[1] < 1 - 1e-9:
        raise ValueError("w_max inviável: w_max·N deve ser ≥ 1")
    ex = rets.sub(cdi.reindex(rets.index), axis=0)
    mu = ex.mean().to_numpy() * TRADING_DAYS
    mu = (1 - shrink) * mu + shrink * mu.mean()
    cov = ex.cov().to_numpy() * TRADING_DAYS
    n = len(mu)

    def neg_sharpe(w):
        var = float(w @ cov @ w)
        return 0.0 if var <= 0 else -float(w @ mu) / np.sqrt(var)

    res = minimize(neg_sharpe, np.full(n, 1.0 / n), method="SLSQP",
                   bounds=[(0.0, w_max)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
                   options={"maxiter": 500, "ftol": 1e-12})
    if not res.success:
        raise RuntimeError(f"otimizador falhou: {res.message}")
    w = pd.Series(np.clip(res.x, 0.0, w_max), index=rets.columns)
    w /= w.sum()
    _check_simplex(w)
    return w


def vol_target_scale(ret, target=0.10, window=21, max_leverage=2.0):
    """Fator diário = alvo / vol realizada (defasado 1 dia), limitado."""
    vol = ret.rolling(window).std(ddof=1) * np.sqrt(252)
    return (target / vol).clip(upper=max_leverage).shift(1).fillna(1.0)