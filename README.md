# Football Player Value Predictor

Projeto de Machine Learning para estimar o valor de mercado de jogadores de
futebol a partir de características básicas de desempenho.

O projeto foi desenvolvido como uma aplicação simples de regressão linear,
utilizando dados reais de jogadores.

## Objetivo

Estimar o valor de mercado de um jogador utilizando cinco características:

- idade;
- posição;
- minutos jogados;
- gols;
- assistências.

## Tecnologias

- Python
- pandas
- NumPy
- scikit-learn
- matplotlib
- joblib

## Estrutura do projeto

```text
football-player-value-predictor/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── reports/
│   └── figures/
│
├── src/
│   ├── prepare_data.py
│   ├── train_model.py
│   └── predict.py
│
├── .gitignore
├── README.md
└── requirements.txt