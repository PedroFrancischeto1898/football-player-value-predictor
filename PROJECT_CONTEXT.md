# Project Context

## Projeto

**Nome:** Football Player Value Predictor

Projeto educacional de Machine Learning para estimar o valor de mercado de
jogadores de futebol a partir de características básicas de desempenho.

O objetivo é manter o projeto simples, interpretável e próximo da ideia original:
coletar/preparar dados, treinar uma regressão linear, avaliar e permitir uma
previsão manual.

---

## Princípio do projeto

**Simplicidade é um requisito.**

Não adicionar novas features, modelos, abstrações ou validações sem uma
necessidade concreta observada no projeto.

Fluxo desejado:

dados → preparação → regressão linear → avaliação → previsão

---

## Dados

Fonte:

`dcaribou/transfermarkt-datasets`

Arquivos utilizados:

- `players.csv.gz`
- `appearances.csv.gz`

Período das estatísticas:

- início: 01/07/2025
- referência: 12/06/2026

Dataset processado:

`data/processed/players_model.csv`

Quantidade de jogadores:

**8.187**

---

## Features

O modelo utiliza somente:

- `age`
- `position`
- `minutes`
- `goals`
- `assists`

Target:

- `market_value_in_eur`

---

## Modelo

Modelo:

`LinearRegression`

Biblioteca:

`scikit-learn`

O primeiro experimento treinou diretamente com valores em euros.

Resultado:

- MAE: €5.648.020
- R²: 0,3462

Foi então testada uma única alteração:

```python
np.log1p(market_value_in_eur)