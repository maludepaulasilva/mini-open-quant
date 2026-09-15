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
