"""Portfólio - ranqueamento, alocação entre estratégias, risco."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from .metrics import TRADING_DAYS, drawdown


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



#RISK

@dataclass(frozen=True)
class DrawdownRule:
    dd_half: float = -0.10
    dd_stop: float = -0.15

    def __post_init__(self):
        if not (self.dd_stop < self.dd_half < 0):
            raise ValueError("exigido dd_stop < dd_half < 0")


FULL, HALF, STOP = 1.0, 0.5, 0.0


def drawdown_control(ret: pd.Series, rule: DrawdownRule = DrawdownRule()) -> pd.Series:
    """Exposição diária ∈ {0, 0.5, 1} conforme o drawdown de `ret`.

    A decisão de t usa o drawdown até t−1, então o resultado já está alinhado ao
    retorno de t — não dê outro shift na hora de aplicar.

    Duas propriedades da regra que convém ter em mente:

    - o gatilho é o drawdown de `ret` *sem* a regra, uma curva-sombra. Não é o
      drawdown de quem segue a regra, logo o max_dd do resultado não fica
      limitado a `dd_stop`;
    - só se volta a FULL num novo topo histórico dessa sombra. Depois de um STOP
      isso significa ficar de fora do repique inteiro, e na volta não se passa
      por HALF (nele só se entra vindo de FULL). Rearmar mais cedo — a um
      drawdown de −5%, digamos — é outra regra, não esta.
    """
    dd = drawdown(ret).shift(1).fillna(0.0)
    exposure = np.empty(len(ret))
    state = FULL
    for t, d in enumerate(dd.to_numpy()):
        if d >= -1e-12:                          state = FULL    # novo topo
        elif d < rule.dd_stop:                   state = STOP
        elif d < rule.dd_half and state == FULL:  state = HALF
        exposure[t] = state
    return pd.Series(exposure, index=ret.index, name="exposure")


def apply_exposure(ret: pd.Series, exposure: pd.Series, cdi: pd.Series) -> pd.Series:
    """Parte exposta segue o portfólio; o resto rende CDI."""
    return exposure * ret + (1 - exposure) * cdi.reindex(ret.index)


def rolling_risk_contribution(rets: pd.DataFrame, w: pd.Series, window: int = 126) -> pd.DataFrame:
    """Fração da variância do portfólio devida a cada estratégia, em janela móvel.

    As frações somam 1 e podem ser negativas: contribuição < 0 é estratégia que
    tira risco do book. Por isso não se plota isso com `.plot.area()`, que
    empilha e exige sinal constante por coluna — veja o notebook 11.
    """
    out = {}
    wv = w.reindex(rets.columns).to_numpy()
    for end in range(window, len(rets) + 1):
        cov = rets.iloc[end - window:end].cov().to_numpy()
        rc = wv * (cov @ wv)
        var = rc.sum()
        out[rets.index[end - 1]] = rc / var if var > 0 else np.full(len(wv), np.nan)
    return pd.DataFrame(out, index=rets.columns).T
