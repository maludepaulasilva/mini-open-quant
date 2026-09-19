"""Reproduz a tabela final do memo: python scripts/run_pipeline.py

Toda linha de `reports/tabela_final.csv` sai daqui, e as duas tabelas da seção 5 do
memo (mais os números da seção 6) são exatamente estas linhas. Sharpe é sempre o
excesso sobre o CDI, anualizado em base 252.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from moq import metrics, portfolio

RAIZ = pathlib.Path(__file__).resolve().parents[1]
DATA = RAIZ / "data" / "processed"

R = pd.read_parquet(DATA / "retornos_oos.parquet")
cdi = pd.read_parquet(DATA / "cdi.parquet")["cdi"].reindex(R.index)
OOS = f"{R.index.min():%Y}–{R.index.max():%Y}"


def vol_target_returns(ret, cdi, k, spread=0.0):
    """A parte alavancada é dinheiro emprestado e paga juros; a sobra rende CDI.

    A versão sem o termo do meio — `k*ret + (1-k)⁺*cdi` — toma emprestado de graça e
    entrega Sharpe +2,19 na paridade de risco. A Fase 6 descartou essa variante.
    """
    return k * ret - (k - 1).clip(lower=0) * (cdi + spread / 252) + (1 - k).clip(lower=0) * cdi


def alocacao_rolante(R, cdi, metodo, anos_treino=3):
    """Pesos treinados em `anos_treino` anos e aplicados no ano seguinte, sem olhar o futuro."""
    anos = sorted(set(R.index.year))
    retornos = []
    for i in range(anos_treino, len(anos)):
        treino = R[R.index.year.isin(anos[i - anos_treino:i])]
        teste = R[R.index.year == anos[i]]
        w = (metodo(treino, cdi.reindex(treino.index), w_max=0.4, shrink=0.5)
             if metodo is portfolio.max_sharpe else metodo(treino))
        retornos.append(portfolio.combine(teste, w))
    return pd.concat(retornos)


linhas = {}


def anota(nome, serie, periodo, nota):
    linhas[nome] = {"periodo": periodo, **metrics.summary(serie, cdi.reindex(serie.index)), "nota": nota}


# 1. as quatro estratégias, fora da amostra, já com 10 bps
for c in R.columns:
    anota(c, R[c], OOS, "estratégia isolada, fora da amostra, 10 bps")

# 2. as três combinações — pesos escolhidos com a amostra toda, sem alavancagem
alloc = {"pesos_iguais": portfolio.equal_weight(R),
         "risk_parity": portfolio.risk_parity(R),
         "max_sharpe": portfolio.max_sharpe(R, cdi, w_max=0.4, shrink=0.5)}
combinacoes = {nome: portfolio.combine(R, w) for nome, w in alloc.items()}
for nome, serie in combinacoes.items():
    anota(nome, serie, OOS, "pesos escolhidos com a amostra toda (otimista)")

# 3. o arquivo final da Fase 6: paridade de risco alavancada, funding cobrado a CDI
k = portfolio.vol_target_scale(combinacoes["risk_parity"], target=0.10, window=21, max_leverage=2.0)
p_final = vol_target_returns(combinacoes["risk_parity"], cdi, k)
salvo = pd.read_parquet(DATA / "portfolio_final.parquet")["portfolio"]
assert np.allclose(p_final, salvo), "divergiu de portfolio_final.parquet"
anota("rp_alavancada_2x", p_final, OOS, f"k no teto de 2,0 em {(k >= 1.999).mean():.1%} dos dias, funding a CDI")

# 4. a versão honesta: pesos refeitos em janela rolante
rolantes = {f"{nome}_rolante": alocacao_rolante(R, cdi, m) for nome, m in
            {"pesos_iguais": portfolio.equal_weight, "risk_parity": portfolio.risk_parity,
             "max_sharpe": portfolio.max_sharpe}.items()}
per_rol = f"{list(rolantes.values())[0].index.min():%Y}–{list(rolantes.values())[0].index.max():%Y}"
for nome, serie in rolantes.items():
    anota(nome, serie, per_rol, "treino de 3 anos, teste de 1, sem olhar o futuro")
idx_rol = list(rolantes.values())[0].index
anota("momentum_mesma_janela", R["momentum"].reindex(idx_rol), per_rol, "referência: a melhor isolada no período do rolante")

# 5. seção 6 do memo: a regra de corte por drawdown sobre a carteira final
for nome, regra in {"rp_alavancada_2x_corte_10_15": portfolio.DrawdownRule(-0.10, -0.15),
                    "rp_alavancada_2x_corte_3_4.5": portfolio.DrawdownRule(-0.03, -0.045)}.items():
    exp = portfolio.drawdown_control(p_final, regra)
    dias = exp.value_counts().reindex([1.0, 0.5, 0.0]).fillna(0).astype(int)
    anota(nome, portfolio.apply_exposure(p_final, exp, cdi), OOS,
          f"dias em cheio/meio/zero: {dias.iloc[0]}/{dias.iloc[1]}/{dias.iloc[2]}")

tabela = pd.DataFrame(linhas).T
print(tabela.drop(columns=["periodo", "nota"]).astype(float).round(3).to_string())
(RAIZ / "reports").mkdir(exist_ok=True)
tabela.to_csv(RAIZ / "reports" / "tabela_final.csv", index_label="serie")
print(f"\nescrito: reports/tabela_final.csv ({len(tabela)} linhas)")
