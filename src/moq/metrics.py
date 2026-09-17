"""Métricas de desempenho. Retornos simples, diários, em decimal. Base 252."""
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def annualized_return(ret: pd.Series) -> float:
    """Composto: (∏(1+r))^(252/n) − 1."""
    n = len(ret)
    if n == 0:
        return float("nan")
    growth = float((1 + ret).prod())
    return growth ** (TRADING_DAYS / n) - 1 if growth > 0 else -1.0


def annualized_vol(ret: pd.Series) -> float:
    """Desvio diário (ddof=1) × √252."""
    return float(ret.std(ddof=1)) * np.sqrt(TRADING_DAYS) if len(ret) > 1 else float("nan")


def sharpe(ret: pd.Series, cdi: pd.Series) -> float:
    """Excesso sobre o CDI, anualizado. NaN se a vol for zero (nunca inf)."""
    ex = ret - cdi.reindex(ret.index)
    sd = float(ex.std(ddof=1)) if len(ex) > 1 else float("nan")
    if not np.isfinite(sd) or sd < 1e-12:
        return float("nan")
    return float(ex.mean()) / sd * np.sqrt(TRADING_DAYS)


def drawdown(ret: pd.Series) -> pd.Series:
    """Série de drawdown, sempre ≤ 0."""
    equity = (1 + ret).cumprod()
    return equity / equity.cummax() - 1


def max_drawdown(ret: pd.Series) -> float:
    return float(drawdown(ret).min()) if len(ret) else float("nan")


def summary(ret: pd.Series, cdi: pd.Series) -> dict:
    """Tudo em um dicionário, para montar tabelas."""
    return {
        "ret_anual": annualized_return(ret),
        "vol_anual": annualized_vol(ret),
        "sharpe": sharpe(ret, cdi),
        "max_dd": max_drawdown(ret),
    }