# Mini Open Quant

Alocador de estratégias sistemáticas para o mercado brasileiro (B3). Projeto de estudo.

## O que faz

4 estratégias (momentum, reversão, par cointegrado, carry DI) → backtest com custos e
walk-forward → ranqueamento → 3 combinações → controle de risco → memo.

**Resultado principal:** com os pesos escolhidos sem olhar o resultado, a melhor combinação
faz Sharpe **+0,08** em 2018-2026, contra **+0,22** do momentum sozinho na mesma janela — e
em 2.163 pregões os dois são indistinguíveis de zero (erro padrão de ±0,34). A recomendação
do memo é **não alocar capital**.

## Como rodar

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest                    # 30 testes
python scripts/run_pipeline.py      # reescreve reports/tabela_final.csv (14 linhas)
```

O pipeline roda offline: lê os parquet já gravados em `data/processed/`. Só os notebooks de
dados baixam algo (Yahoo para preços, SGS/BCB para o CDI).

## Estrutura

```
src/moq/     data.py · metrics.py · strategies.py · backtest.py · portfolio.py
notebooks/   04.1_dados · 06_estrategias · 08_backtest · 10_portfolio · 12_risco (nessa ordem)
tests/       um arquivo por módulo
reports/     memo_comite.md, tabela_final.csv
scripts/     run_pipeline.py — toda linha da tabela final sai daqui
data/        raw/ (o que foi baixado) · processed/ (o que as fases consomem)
```

Os notebooks de número ímpar (`03`, `05`, `07`, `09`, `11`) são os rascunhos de cada fase; os
`_reordenado`/`.1` são as versões finais, e é a elas que o memo se refere.

## Limitações

1. **Viés de sobrevivência** — os 23 tickers foram escolhidos hoje, entre quem continua grande e líquido; o erro tem sempre o mesmo sinal e contamina principalmente o momentum.
2. **Custo incompleto** — 10 bps sobre Σ|Δw| tratam compra e venda igual: não há aluguel na perna vendida nem impacto de mercado (a reversão gira 254×/ano).
3. **Giro entre estratégias não é cobrado** — a combinação rebalanceia diariamente e o vol targeting muda o tamanho todo dia, ambos de graça.
4. **`w_max` é a alocação, não o otimizador** — com 1,0 sai 100% momentum, com 0,25 saem pesos iguais; e a covariância é de amostra única num regime que não é fixo.
5. **Um só recorte** — um par, um ativo de carry, um universo, um país. Sem intervalo de confiança ao lado do Sharpe, nada disso é conclusão.

Detalhe e números em [`reports/memo_comite.md`](reports/memo_comite.md).
