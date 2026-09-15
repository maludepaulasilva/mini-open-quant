"""Dados - validação, limpeza, alinhamento."""

import logging
import pandas as pd

log = logging.getLogger(__name__)

MAX_MISSING_FRACTION = 0.05
FFILL_LIMIT = 3


def validate_prices(prices: pd.DataFrame) -> None:
    """Lança exceção se os preços violares as regras básicas. não altera nada."""
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices deve ser DataFrame (datas x ativos)")
    idx = prices.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise TypeError("índice deve ser DatetimeIndex")
    if not idx.is_monotonic_increasing:
        raise ValueError("índice de datas fora de ordem")
    if idx.has_duplicates:
        raise ValueError("datas duplicadas")
    if (prices <=0).any().any():
        raise ValueError("preço não positivo encontrado")


def clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Remove ativos com dados demais faltando e preenche lacunas curtas."""
    validate_prices(prices)

    frac = prices.isna().mean()
    keep = frac[frac <= MAX_MISSING_FRACTION].index
    dropped = frac[frac > MAX_MISSING_FRACTION].index
    for t in dropped:
        log.warning("ativo %s removido: %.1f%% ausente", t, 100 * frac[t])

    return prices[keep].ffill(limit=FFILL_LIMIT)


def fetch_cdi(start: str, end: str | None = None, years_per_chunk: int = 5) -> pd.Series:
    """Baixa o CDI diário (SGS 12) em blocos e concatena. Retorna em decimal, não em %.

    O SGS recusa consultas diárias com janela maior que 10 anos.
    """
    from bcb import sgs

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()

    chunks = []
    chunk_start = start_ts
    while chunk_start <= end_ts:
        chunk_end = min(chunk_start + pd.DateOffset(years=years_per_chunk) - pd.Timedelta(days=1), end_ts)
        chunks.append(sgs.get({"cdi": 12}, start=chunk_start, end=chunk_end)["cdi"])
        chunk_start = chunk_end + pd.Timedelta(days=1)

    cdi = pd.concat(chunks).sort_index()
    cdi = cdi[~cdi.index.duplicated(keep="last")]
    cdi.index = pd.DatetimeIndex(cdi.index)
    return (cdi / 100).astype(float)


def align_cdi(cdi_daily: pd.Series, prices: pd.DataFrame) -> pd.Series:
    """Coloca o CDI no mesmo calendário dos preços. Dias sem CDI viram 0."""
    if(cdi_daily > 0.01).any():
        raise ValueError("CDI diário > 1%: a série está em % - divida por 100")
    aligned = cdi_daily.reindex(prices.index)
    n_missing = int(aligned.isna().sum())
    if n_missing:
        log.warning("CDI ausente em %d pregões; preenchido com 0", n_missing)
    return aligned.fillna(0.0)
