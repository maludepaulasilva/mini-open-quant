"""Motor de backtest — um só, para qualquer estratégia que respeite o contrato."""
import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

# a mesma definição de Sharpe da tabela de métricas — uma só, para não divergirem
from .metrics import sharpe as _sharpe

@dataclass(frozen=True)
class BacktestResult:
    ret: pd.Series
    gross: pd.Series
    cost:  pd.Series
    cash: pd.Series
    turnover: pd.Series
    cost_bps: float


def run(weights: pd.DataFrame, prices: pd.DataFrame, cdi: pd.Series,
        cost_bps: float = 10.0) -> BacktestResult:
     """ret_t = Σ w_t·r_t  −  Σ|Δw_t|·bps/1e4  +  (1 − Σ|w_t|)⁺·cdi_t"""
     if cost_bps < 0:
        raise ValueError("cost_bps deve ser >= 0")
     if not weights.index.equals(prices.index):
        raise ValueError("índice dos pesos difere do índice dos preços")
     if list(weights.columns) != list(prices.columns):
        raise ValueError("colunas dos pesos diferem das dos preços")
     if weights.isna().any().any():
        raise ValueError("pesos com NaN")
     cdi_al = cdi.reindex(prices.index)
     if cdi_al.isna().any():
        raise ValueError("CDI não cobre todo o índice de preços")

     rets = prices.pct_change(fill_method=None).fillna(0.0)
     gross = (weights * rets).sum(axis=1)
     to = weights.diff().abs().sum(axis=1)
     to.iloc[0] = weights.iloc[0].abs().sum()
     cost = to * cost_bps / 1e4
     cash = (1.0 - weights.abs().sum(axis=1)).clip(lower=0.0) * cdi_al
     net = gross - cost + cash
     return BacktestResult(ret=net, gross=gross, cost=cost, cash=cash, turnover=to, cost_bps=cost_bps)


@dataclass(frozen=True)
class Fold:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    chosen: dict
    train_sharpe: float
    test_sharpe: float


def _grid(space: dict) -> list[dict]:
    """Produto cartesiano do espaço de busca: {"a": [1, 2], "b": [3]} → [{a:1,b:3}, {a:2,b:3}]."""
    keys = list(space)
    return [dict(zip(keys, v)) for v in itertools.product(*(space[k] for k in keys))]


def walk_forward(strategy, prices: pd.DataFrame, cdi: pd.Series, space: dict,
                 train_years: int = 3, test_years: int = 1,
                 cost_bps: float = 10.0, warmup: int = 0):
    """Escolhe parâmetros no treino, avalia no teste seguinte; devolve (ret fora da amostra, folds).

    O teste recebe `warmup` pregões anteriores para LER (encher as janelas das estratégias),
    mas só os dias do teste entram na avaliação. Dimensione warmup pelo maior parâmetro de
    janela da grade — momentum precisa de lookback + skip, pares de 2 × window.
    """
    years = sorted(set(prices.index.year))
    if len(years) < train_years + test_years:
        raise ValueError("histórico insuficiente para um fold")
    grid = _grid(space)
    oos, folds = [], []

    for i in range(train_years, len(years) - test_years + 1):
        tr_years = years[i - train_years:i]
        te_years = years[i:i + test_years]
        tr = prices[prices.index.year.isin(tr_years)]
        te_mask = prices.index.year.isin(te_years)
        te = prices[te_mask]
        start = int(np.argmax(te_mask))
        te_warm = prices.iloc[max(0, start - warmup): start + int(te_mask.sum())]
        assert tr.index.max() < te.index.min(), "treino e teste se sobrepõem"

        scores = {}

        def score(p):
            chave = tuple(sorted(p.items()))
            if chave not in scores:
                s = _sharpe(run(strategy(tr, **p), tr, cdi, cost_bps).ret, cdi)
                scores[chave] = -np.inf if not np.isfinite(s) else s
            return scores[chave]

        chosen = max(grid, key=score)
        w_te = strategy(te_warm, **chosen).loc[te.index]
        r_te = run(w_te, te, cdi, cost_bps).ret
        folds.append(Fold(tr.index.min(), tr.index.max(), te.index.min(), te.index.max(),
                          chosen, score(chosen), _sharpe(r_te, cdi)))
        oos.append(r_te)

    out = pd.concat(oos)
    assert out.index.is_unique
    return out, folds
