"""Estratégias - cada uma devolve um DataFrame de pesos (datas x ativos)."""

import functools
import numpy as np
import pandas as pd

_TOL = 1e-9

def _postconditions(weights: pd.DataFrame, prices: pd.DataFrame, max_gross: float) -> None:
    """Verifica o que toda estratégia promete. Lança AssertionError se quebrar."""
    if not weights.index.equals(prices.index):
        raise AssertionError("índice dos pesos difere do índice dos preços")
    if list(weights.columns) != list(prices.columns):
        raise AssertionError("coluna dos pesos diferem das colunas dos preços")
    if weights.isna().any().any():
        raise AssertionError("pesos contêm NaN")
    gross = weights.abs().sum(axis=1)
    if (gross > max_gross + _TOL).any():
        raise AssertionError(f"exposição bruta máxima {gross.max():4f} > {max_gross}")


def lagged(max_gross: float = 1.0):
    """Decorador: defasa os pesos em 1 dia e verifica o contrato."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(prices, *args, **kwargs):
            raw = fn(prices, *args, **kwargs)
            w = (raw.reindex(index=prices.index, columns=prices.columns)
                 .shift(1)
                 .fillna(0.0))
            _postconditions(w, prices, max_gross)
            return w
        wrapper.__moq_lagged__ = True
        return wrapper
    return deco


def _normalize_rows(w: pd.DataFrame) -> pd.DataFrame:
    """Faz cada linha somar Σ|w| = 1. Linhas vazias ficam o (não NaN)."""
    gross = w.abs().sum(axis=1)
    return w.div(gross.where(gross > 0, np.nan), axis=0).fillna(0.0)

@lagged(max_gross=1.0)
def momentum(prices, lookback=252, skip=21, top=0.2, rebalance="ME"):
    """Compara o top 'top' por retorno entre t-lookback e t-skip. Long-only, rebalanceio mensal"""
    if not (lookback > skip > 0):
        raise ValueError("exigido lookback > skip > 0")
    if not (0 < top <= 1):
        raise ValueError("exigido 0 < top <= 1")
    mom = prices.shift(skip) / prices.shift(lookback) - 1.0
    rank = mom.rank(axis=1, pct=True)
    sel = (rank >= 1.0 - top).astype(float).where(mom.notna(), 0.0)
    sel = _normalize_rows(sel)
    return sel.resample(rebalance).last().reindex(prices.index, method="ffill").fillna(0.0)
    

@lagged(max_gross=1.0)
def mean_reversion(prices, short=5, long=60, threshold=1.5):
    """Contra o z-score do retorno de 'short' dias (janela de 'long'). Long-short, Σ|w| = 1."""
    if not (long > short > 0):
        raise ValueError("exigido long > short > 0")
    r = prices.pct_change(short, fill_method=None)
    z = (r - r.rolling(long).mean()) / r.rolling(long).std(ddof=1)
    signal = (-z).where(z.abs() > threshold, 0.0).fillna(0.0)
    return _normalize_rows(signal)

@lagged(max_gross=1.0)
def pairs(prices, a, b, window=252, entry=2.0, exit=0.5, stop=4.0):
     """Spread = log(Pa) − β·log(Pb). Entra em |z|>entry, sai em |z|<exit, stop em |z|>stop."""
     if not (stop > entry > exit >= 0):
         raise ValueError("exigido stop > entry > exit >= 0")
     la, lb = np.log(prices[a]), np.log(prices[b])
     beta = la.rolling(window).cov(lb) / lb.rolling(window).var(ddof=1)
     spread = la - beta * lb
     z = (spread - spread.rolling(window).mean()) / spread.rolling(window).std(ddof=1)

     state, pos, zv = 0, np.zeros(len(z)), z.to_numpy()
     for t in range(len(zv)):
         if np.isnan(zv[t]):
            state = 0
         elif state == 0:
            if zv[t] > entry : state = -1
            elif zv[t] < -entry: state = 1
         elif abs(zv[t]) < exit or abs(zv[t]) > stop:
            state = 0
         pos[t] = state

     w = pd.DataFrame(0.0, index = prices.index, columns=prices.columns)
     w[a] = pos * 0.5
     w[b] = -pos * beta.fillna(0.0).clip(-1, 1) * 0.5
     return _normalize_rows(w).where(w.abs().sum(axis=1) > 0, 0.0)

def prefixed_bond_return(y_1y, cdi):
    """r_t ≈ y_{t-1}/252 − D·(y_t − y_{t-1}), D = 1/(1+y_{t-1}). Sem NaN."""
    y = y_1y.reindex(cdi.index).ffill()
    dur = 1.0 / (1.0 + y.shift(1))
    return (y.shift(1) / 252.0 - dur * y.diff()).fillna(0.0)

def carry_signal(y_1y, cdi, window=252, k=1.0):
    """1 quando a inclinação (y − CDI anual) excede média móvel + k·desvio. SEM defasagem."""
    cdi_annual = (1.0 + cdi) ** 252 - 1.0
    slope = y_1y.reindex(cdi.index).ffill() - cdi_annual
    thr = slope.rolling(window).mean() + k * slope.rolling(window).std(ddof=1)
    return (slope > thr).astype(float).fillna(0.0)

def carry_returns(y_1y, cdi, **kw):
    """Retorno diário: no pré quando o sinal de ontem está ligado, senão no CDI."""
    sig = carry_signal(y_1y, cdi, **kw).shift(1).fillna(0.0)
    return sig * prefixed_bond_return(y_1y, cdi) + (1.0 - sig) * cdi



