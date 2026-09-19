# Memo - Mini Open Quant: alocador de estratégias sistemáticas na B3

**Para:** Comitê de investimentos · **De:** Maria Luisa · **Data:** 19/09/2026

## 1. Resumo executivo

Combinar as quatro estratégias não bateu a melhor estratégia isolada, e nenhuma das duas bate o CDI de forma distinguível de zero.

- **O que foi feito**.
Um alocador de quatro estratégias sistemáticas - momentum, reversão à média, par cointegrado e carry na curva DI - sobre 23 ações da B3 e o pré de 1 ano, de 2015 a 2026, com 10 bps de custo por unidade de peso negociada e parâmetros escolhidos por walk-forward.

- **O resultado, em um número**.
Com os pesos escolhidos sem olhar o resultado (treino de 3 anos, teste de 1 ano), a melhor combinação entrega Sharpe +0,08 em 2018-2026; o momentum isolado, na mesma janela, entrega +0,22. Em 2.163 pregões, os dois são estatisticamente indistinguíveis de zero - e o momentum só é "a melhor isolada" depois de eu ter visto o resultado das quatro.

- **Recomendação: não alocar capital**.
A tese de diversificação se confirmou (correlação média de 0,02 entre as estratégias, vol da carteira de paridade de risco em 1,8% ao ano), mas diversificação sem retorno esperado positivo produz um CDI caro. O único componente com Sharpe positivo é o momentum, e é justamente o mais contaminado pela limitação abaixo. O que eu recomendo é reconstruir o universo com o CATAHIST da B3 e repetir a Fase 5 antes de qualquer decisão de alocação.

- **Maior limitação.**
Viés de sobrevivência: os 23 tickers foram escolhidos hoje, entre os que continuam grandes e líquidos. Informação do futuro entrou na seleção do passado, o erro tem sempre o mesmo sinal e não sai com mais dados nem com validação cruzada.

## 2. Universo e dados

23 ações da B3 com série completa de 2012 a 2026, sem um único NaN depois da limpeza. Partiram de 30 tickers escritos à mão: as maiores e mais líquidas do Ibovespa, com setores repetidos de propósito porque a estratégia de pares precisa de candidatos parecidos.

As 7 remoções são de dois tipos, e só um é problema: ELET3, EMBR3 e JBSS3 voltaram vazias do provedor (100% ausente), e BBSE3, RAIL3, BPAC11 e HAPV3 existem, mas não desde 2012 - a regra de 5% de dados faltantes derruba as quatro porque a janela começa em 2012. O efeito é um universo que não muda de tamanho: 23 ativos em todos os 3.653 pregões. Um universo honesto deveria crescer conforme as empresas abriam capital; o meu não cresce porque foi recortado para não crescer.

As outras três séries: **CDI** da série 12 do SGS/BCB, em decimal (2023 acumula 13,04% na série original e 12,99% alinhado ao calendário da B3 - a diferença de 0,05 p.p. empurra sempre a favor da estratégia); **curva pré de 1 ano** do Tesouro Transparente, tomando a cada dia o Tesouro Prefixado com prazo mais próximo de 365 dias corridos (mediana de 67 dias de distância do alvo, p90 de 163); e os **fatores do NEFIN/USP** (MKT, SMB, HML, WML), que não entram em estratégia nenhuma e servem só de régua. Vale registrar o que o MKT diz do período: de 2012 a 2026 o mercado brasileiro acumulou **−7% acima do CDI**. O benchmark deste memo é duro por construção.

## 3. Estratégias

Cada estratégia devolve uma matriz de pesos (datas × ativos), defasada em 1 dia por decorador e com exposição bruta Σ|w| ≤ 1 verificada por asserção - nada aqui usa o preço do próprio dia. O custo de 10 bps incide sobre Σ|Δw| de cada pregão.

| estratégia | o que faz | parâmetros | giro/ano | breakeven de custo |
|---|---|---|---|---|
| **momentum** | compra o top 20-30% por retorno de 6-12 meses, pulando o último mês; long-only, rebalanceio mensal | `lookback` ∈ {126, 252}, `skip`=21, `top` ∈ {0,2; 0,3} | 5,3× | **238 bps** |
| **reversão** | posição contra o z-score do retorno de 5-10 dias (janela de 60); long-short com Σ|w| = 1 | `short` ∈ {5, 10}, `long`=60, `threshold` ∈ {1,5; 2,0} | **254×** | **1,6 bps** |
| **pares** | spread log(ITUB4) − β·log(BBDC4) com β rolante; entra em \|z\| > entry, sai em \|z\| < 0,5, stop em \|z\| > 4 | `entry` ∈ {1,5; 2,0}, `window` ∈ {126, 252} | 3,0× | **−166 bps** |
| **carry** | fica no pré de 1 ano quando a inclinação (pré − CDI anual) passa a média móvel + 1 desvio; senão, CDI | `window`=252, `k`=1,0 (sem grade) | — | — |

Duas coisas que a coluna de breakeven já entrega, antes de qualquer discussão de Sharpe:

- **A reversão não é uma estratégia cara, é uma estratégia que só existe em planilha sem custo.** Ela gira a carteira 254 vezes por ano e empata com o CDI a **1,6 bps** de custo. Com os 10 bps do projeto o Sharpe vai de +0,13 para −0,68; com 30 bps, para −2,30.
- **O par nunca passou no teste de cointegração** (p = 0,43), e o breakeven negativo diz a mesma coisa por outro caminho: ele perde do CDI de graça, sem pagar corretagem nenhuma.

O carry merece um registro à parte porque a leitura dele mudou quando a curva simulada virou curva real: **o pré de 1 ano só está acima do CDI em 51,6% dos dias**, com inclinação média de +0,17 p.p. Não existe prêmio de prazo permanente nessa ponta - a inclinação é fortemente positiva quando o mercado precifica alta da Selic (2021: +2,6 p.p.) e negativa quando precifica corte (2016, 2017, 2023: entre −0,9 e −1,4 p.p.). O carry, portanto, não captura prêmio de risco: aposta que a expectativa embutida na curva está errada para cima, e o sinal por média móvel liga justamente perto do topo do ciclo.

## 4. Validação

O motor é uma conta só, igual para as quatro: `r = Σ w·r − Σ|Δw|·bps/1e4 + (1 − Σ|w|)⁺·cdi`. Duas defesas foram montadas em cima dela.

**Primeira: medir o tamanho do acaso.** Varri 60 combinações de parâmetros do momentum num mercado sintético sem sinal nenhum - 20 ativos em passeio aleatório puro. A melhor das 60 deu **Sharpe +0,73**, a mediana +0,15 e a pior −0,44. Quase 1,2 de Sharpe de distância entre a melhor e a pior configuração, tudo ruído, por construção. Essa é a régua com que o resto do memo tem que ser lido: um Sharpe positivo só informa junto com a resposta a "de quantas tentativas ele é o melhor?".

**Segunda: escolher num lugar, medir em outro.** Walk-forward de treino de 3 anos e teste de 1, andando um ano por vez - 12 folds, grades de quatro combinações cada, escritas antes de rodar e não mexidas depois. O teste recebe pregões anteriores só para encher as janelas (`warmup` dimensionado pela maior janela da grade: 300 no momentum, 505 nos pares); só os dias do ano de teste entram na conta.

| estratégia | Sharpe médio no treino | Sharpe médio no teste | folds em que caiu |
|---|---|---|---|
| momentum | +0,48 | **+0,25** | 8/12 |
| reversão | −0,10 | **−0,92** | 10/12 |
| pares | −0,30 | **−0,80** | 9/12 |

A queda do treino para o teste é a medida do otimismo que havia antes. Mas o achado mais informativo não está nos números: está nos **parâmetros escolhidos, que não estabilizam**. No momentum o treino pediu `lookback`=252 em 7 folds e 126 em 5; `top`=0,2 em 7 e 0,3 em 5. É cara ou coroa. Se a vantagem de um parâmetro fosse real, ele venceria de forma consistente ano após ano.

Nenhum fold tem Sharpe de treino absurdo (o maior é +1,34, em 2020), o que afasta suspeita de vazamento de dado: o problema é falta de sinal, não look-ahead. O carry não passa pelo walk-forward porque não tem grade - nada nele foi escolhido olhando resultado, então não há o que tirar da amostra.

## 5. Resultados fora da amostra

**As quatro isoladas**, 2015-2026, 2.912 pregões, já com 10 bps. CDI no período: 10,0% ao ano.

| série | ret. anual | vol anual | Sharpe | max DD | corr. média |
|---|---|---|---|---|---|
| momentum | **16,6%** | 26,6% | **+0,35** | −54,1% | +0,03 |
| carry | 9,9% | 1,0% | −0,08 | −2,5% | +0,02 |
| pares | 3,7% | 8,9% | −0,62 | −17,6% | −0,03 |
| reversão | −21,3% | 34,3% | −0,79 | −95,4% | +0,05 |

A boa não é a diferente. A única acima do CDI é o momentum, e é caríssima em risco (vol de 26,6%, drawdown de −54%); a única com correlação média negativa é o par, que rende 3,7% ao ano; a mais mansa é o carry, que é quase caixa com ruído. A reversão perde em tudo e ainda é a de maior correlação média.

A degradação de cada uma, na mesma régua (Sharpe do excesso sobre o CDI):

| | sem custo | amostra toda, 10 bps | fora da amostra, 10 bps |
|---|---|---|---|
| momentum | +0,50 | +0,48 | **+0,35** |
| reversão | +0,13 | −0,68 | **−0,79** |
| pares | −0,60 | −0,63 | **−0,62** |
| carry | −0,09 | — | **−0,08** |

**As três combinações.** Com os pesos calculados na amostra inteira - isto é, espiando o gabarito - e depois com os pesos refeitos em janela rolante de 3 anos, que é a versão em que eu acredito:

| | pesos da amostra toda (2015-26) | mesmo, recortado em 2018-26 | **pesos rolantes (2018-26)** |
|---|---|---|---|
| pesos iguais | −0,50 | −0,57 | **−0,57** |
| paridade de risco | −0,55 | −0,67 | **−0,74** |
| máximo Sharpe | **+0,25** | +0,14 | **+0,08** |
| melhor isolada (momentum) | +0,35 | +0,22 | +0,22 |

O `max_sharpe` perde **dois terços** do resultado quando para de espiar (0,25 → 0,08), e a coluna do meio separa as duas causas: metade é só o período, metade é o gabarito. Ele era, aliás, o único método acima do CDI - e é o único que usa médias históricas para decidir peso. Pesos iguais não muda nada entre as duas últimas colunas, como tem que ser: não usa dado nenhum, logo não existe versão dele fora da amostra.

**O tamanho do erro.** Para Sharpe anualizado, o erro padrão é da ordem de √(252/n): em 2.163 pregões, **±0,34**. Os dois números que sobraram - +0,08 da melhor combinação e +0,22 do momentum - ficam a 0,2 e 0,6 erro padrão de zero. Não são resultados pequenos: são resultados ausentes.

**E o retorno é do mercado, não meu.** Na regressão contra os quatro fatores do NEFIN, **nenhum alpha positivo em lugar nenhum da tabela**. O momentum tem alpha de −1,3% ao ano (t = −0,28) com β<sub>MKT</sub> = 0,94, β<sub>WML</sub> = 0,35 e R² = 0,66: comprar as ações que mais subiram é estar quase inteiramente comprado no mercado, e o Sharpe de +0,35 é beta, que se compra num ETF sem backtest nenhum. O `max_sharpe` é uma versão desalavancada disso (alpha −1,7%, t = −0,87, β<sub>MKT</sub> = 0,36). Reversão (−31,1%, t = −3,07) e pares (−5,7%, t = −2,19) têm alpha negativo com significância; só o carry tem alpha indistinguível de zero por mérito, e não por falta de dado.

## 6. Risco

A carteira arquivada em `data/processed/portfolio_final.parquet` é a paridade de risco alavancada em 2×, com o empréstimo cobrado a CDI: **7,7% ao ano, vol de 3,4%, drawdown máximo de −5,7%, Sharpe −0,60**. Cinco observações sobre ela.

- **A mesma carteira sem cobrar o funding daria Sharpe +2,19.** É o número mais bonito do projeto e é integralmente artificial: a fórmula ingênua remunera a sobra quando `k` < 1 e não cobra nada quando `k` > 1, ou seja, toma emprestado a juro zero. A CDI + 1% ao ano - ainda otimista para pessoa física - o Sharpe é −0,89. Alavancar carteira que rende menos que o custo do dinheiro só multiplica o prejuízo.
- **Não há vol targeting nenhum aqui.** O alvo de 10% exigiria 5,6× de alavancagem sobre uma carteira de vol 1,8%, e o teto é 2: `k` fica grudado em 2,0 em **98,2% dos pregões** e nunca cai abaixo de 1. É uma posição alavancada em 2× de ponta a ponta com o nome errado. Para o mecanismo operar, o alvo teria que ser compatível com a carteira - algo como 3%.
- **A regra de corte por drawdown é inerte nos limiares de livro.** Com −10%/−15% ela nunca dispara: os 2.912 pregões ficam em exposição cheia, porque o pior drawdown da carteira é −5,7%. Recalibrando para −3%/−4,5% - calibragem à posteriori, leia como ilustração - a regra passa a atuar em 342 dias, o drawdown máximo cai para −4,5% e o Sharpe piora de −0,60 para **−0,86**: a parada trava o prejuízo e perde o repique. E nenhuma troca de estado paga corretagem no modelo.
- **Peso em dinheiro não é peso em risco.** Com 25% em cada, a reversão ocupava **60% do risco** da carteira e o carry 0% - "pesos iguais" era, na prática, uma aposta na pior das quatro. A paridade de risco iguala de fato as frações (22% a 27% cada), mas o preço é colocar **84,4% do dinheiro no carry** e 3,3% no momentum.
- **Choques.** Beta de mercado da carteira final: 0,055, medido direto na série alavancada - fazer Σ w·β sem a alavancagem subestima por ~2×. Bolsa −30% em um dia ≈ **−1,6%** na carteira. Juros +300 bps em um dia ≈ **−1,6%** (β de −0,31 por 100% de Δy, R² = 0,33, y de 13,68% hoje). Se todas as correlações fossem a 1, a vol iria de 3,4% para 6,9% - é esse o tamanho do ganho de diversificação.

E a ressalva que vale mais que as cinco: **a diversificação some no dia em que ela serviria**. Nos 5% piores dias do mercado (143 pregões), a correlação momentum × reversão vai de +0,15 para **+0,60**, e o par, que parecia hedge por ser o único com correlação negativa, troca de sinal (−0,10 → +0,10). As duas estratégias que carregam risco de verdade são independentes nos dias normais e viram a mesma aposta no dia ruim. O que continua descorrelacionado no estresse é o carry - e o mérito dele é não ter comportamento nenhum.

## 7. O que faria diferente (5 pontos honestos)

1. **Reconstruir o universo ano a ano com o COTAHIST da B3.** É o conserto mais importante e o único que não é um ajuste de modelo: enquanto os tickers forem escolhidos entre os sobreviventes de hoje, todo resultado é um teto, o erro tem sempre o mesmo sinal, e o mais contaminado é justamente o momentum - a única estratégia que sobrou com Sharpe positivo.
2. **Cobrar o que as pernas vendidas custam.** O modelo trata compra e venda igual, a 10 bps sobre Σ|Δw|. Vender exige alugar o papel (de meio a vários por cento ao ano, muito mais em papel difícil), e o custo linear ignora impacto de mercado - que cresce com o tamanho da ordem relativo ao volume, dado que esta base nem tem. Pares e a perna curta da reversão pioram, e a reversão, com 254 giros por ano, é a mais exposta.
3. **Cobrar também o giro *entre* as estratégias.** Os 10 bps cobrem o rebalanceamento dentro de cada estratégia. A combinação assume pesos fixos com rebalanceamento diário implícito, e o vol targeting muda o tamanho da posição todo dia - nenhuma das duas coisas paga nada. Faltou uma coluna de turnover na tabela de combinações.
4. **Aposentar o `max_sharpe` como está e reportar só a janela rolante.** A seção 5 mostra que `w_max` **é** a alocação (1,0 dá 100% momentum, 0,25 dá pesos iguais, e nos casos intermediários pelo menos duas estratégias ficam grudadas no teto), e que o `shrink` não muda nada até mudar tudo (em 0,75 a reversão salta de 0 para 40%). Pesos que dependem assim de um parâmetro que não tenho como calibrar não são resposta de otimizador, são número digitado. Junto com isso: covariância de amostra única num regime que não é fixo.
5. **Testar em outro recorte antes de chamar qualquer coisa de conclusão.** Um par, um ativo de carry, um universo, um país. E reportar o intervalo de confiança do Sharpe ao lado do Sharpe - a seção 5 mostra que, com ±0,34 de erro padrão, quase tudo que este projeto produziu cabe dentro de zero.
