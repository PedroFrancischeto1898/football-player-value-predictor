# Model Card — Football Player Value Predictor

## 1. Visão geral

O **Football Player Value Predictor** é um projeto educacional de Machine
Learning desenvolvido para estimar o valor de mercado de jogadores de futebol
a partir de um conjunto pequeno de características individuais.

O projeto utiliza uma regressão linear e foi deliberadamente mantido simples,
com foco em:

- preparação de dados;
- treinamento de modelo;
- avaliação quantitativa;
- análise de erros;
- interpretação de resultados;
- documentação;
- disponibilização de uma previsão utilizável.

O modelo não pretende reproduzir sistemas profissionais de valuation de atletas.

---

## 2. Objetivo do modelo

O objetivo é estimar:

```text
market_value_in_eur
```

a partir das seguintes características:

- `age`
- `position`
- `minutes`
- `goals`
- `assists`

O modelo procura responder à seguinte pergunta:

> Dadas a idade, posição e estatísticas básicas de desempenho de um jogador,
> qual valor de mercado pode ser estimado a partir dos padrões presentes nos
> dados utilizados para treinamento?

---

## 3. Dados

Os dados utilizados são provenientes do projeto público:

`dcaribou/transfermarkt-datasets`

Arquivos utilizados:

- `players.csv.gz`
- `appearances.csv.gz`

As estatísticas dos jogadores foram agregadas no período entre:

- início: 01/07/2025
- data de referência: 12/06/2026

Após preparação e filtragem, o dataset final contém:

**8.187 jogadores**

---

## 4. Variáveis utilizadas

### Variáveis de entrada

| Variável | Descrição |
|---|---|
| `age` | Idade do jogador na data de referência |
| `position` | Posição geral do jogador |
| `minutes` | Total de minutos jogados no período |
| `goals` | Total de gols |
| `assists` | Total de assistências |

### Variável alvo

```text
market_value_in_eur
```

Representa o valor de mercado do jogador em euros.

---

## 5. Preparação dos dados

O pipeline de preparação realiza:

1. carregamento dos dados de jogadores;
2. carregamento dos registros de aparições;
3. filtragem pelo período de interesse;
4. agregação de minutos, gols e assistências por jogador;
5. cálculo da idade;
6. junção das informações dos jogadores com as estatísticas;
7. remoção de registros incompletos;
8. remoção de valores de mercado não positivos;
9. geração do dataset final utilizado pelo modelo.

O arquivo processado é:

```text
data/processed/players_model.csv
```

---

## 6. Modelo

O algoritmo utilizado é:

```python
LinearRegression()
```

da biblioteca `scikit-learn`.

A variável categórica `position` é transformada utilizando:

```python
OneHotEncoder(handle_unknown="ignore")
```

As demais variáveis são utilizadas diretamente pela regressão.

---

## 7. Transformação do target

O primeiro experimento utilizou diretamente:

```text
market_value_in_eur
```

Esse modelo apresentou:

- MAE: €5.648.020
- R²: 0,3462

A análise gráfica mostrou:

- concentração de jogadores em valores baixos;
- subestimação de jogadores de alto valor;
- ocorrência de previsões negativas.

Foi então testada uma transformação logarítmica:

```python
np.log1p(market_value_in_eur)
```

Após a previsão, o valor retorna à escala original utilizando:

```python
np.expm1(prediction)
```

O modelo com target logarítmico apresentou:

- MAE: €4.276.655
- R² em euros: 0,2867

Embora o R² em euros tenha diminuído, o MAE foi reduzido em aproximadamente
24% em relação ao primeiro modelo.

Por esse motivo, o modelo com target logarítmico foi mantido no MVP.

---

## 8. Divisão treino/teste

O dataset foi dividido em:

- 80% para treinamento;
- 20% para teste.

Foi utilizado:

```python
random_state=42
```

para permitir a reprodução da mesma divisão.

Quantidade de observações:

| Conjunto | Jogadores |
|---|---:|
| Treino | 6.549 |
| Teste | 1.638 |
| Total | 8.187 |

---

## 9. Baseline

Para verificar se o modelo realmente aprende informação útil, foi criado um
baseline simples.

O baseline prevê, para todos os jogadores, a mediana do valor de mercado
observada apenas no conjunto de treinamento.

Valor utilizado pelo baseline:

**€1.500.000**

Resultados:

| Modelo | MAE |
|---|---:|
| Baseline pela mediana | €5.195.650 |
| Regressão linear | €4.276.655 |

O modelo apresentou uma redução de:

**17,7% no MAE em relação ao baseline.**

Isso indica que as características utilizadas pelo modelo fornecem informação
preditiva além de uma estimativa constante simples.

---

## 10. Distribuição dos erros

Além do MAE global, foi analisada a distribuição dos erros absolutos.

| Percentil | Erro absoluto |
|---|---:|
| 50% | até €1.082.526 |
| 80% | até €4.735.271 |
| 90% | até €11.976.072 |

O erro absoluto mediano foi:

**€1.082.526**

Portanto, metade das previsões realizadas no conjunto de teste apresentou erro
absoluto de até aproximadamente €1,08 milhão.

Esses valores representam a distribuição observada dos erros no conjunto de
teste e não constituem intervalos probabilísticos de confiança.

---

## 11. Desempenho por faixa de valor

A avaliação foi também realizada separadamente por faixa de valor de mercado.

| Valor real | Jogadores | MAE | Erro mediano |
|---|---:|---:|---:|
| < €5M | 1.202 | €1.146.089 | €692.288 |
| €5M–€20M | 296 | €6.161.646 | €5.147.812 |
| €20M–€50M | 109 | €20.762.384 | €19.712.915 |
| ≥ €50M | 31 | €49.697.238 | €51.598.669 |

A análise mostra uma diferença significativa de desempenho entre as faixas.

O modelo apresenta desempenho consideravelmente melhor entre jogadores com
valor inferior a €5 milhões.

À medida que o valor de mercado aumenta, o erro também aumenta de forma
substancial.

Jogadores de valor muito elevado tendem a ser subestimados.

---

## 12. Principais resultados

Os principais achados desta versão são:

- o modelo supera um baseline simples;
- o MAE é 17,7% menor que o baseline;
- metade das previsões possui erro absoluto inferior a aproximadamente €1,08M;
- o desempenho é significativamente melhor para jogadores abaixo de €5M;
- jogadores de alto valor apresentam erros substancialmente maiores;
- uma única métrica global não representa igualmente bem todas as faixas de valor.

---

## 13. Limitações

O modelo utiliza apenas cinco características.

Ele não considera fatores relevantes para valuation de jogadores, entre eles:

- clube atual;
- qualidade da liga;
- duração de contrato;
- potencial futuro;
- reputação;
- seleção nacional;
- histórico de lesões;
- situação financeira dos clubes;
- demanda de mercado;
- contexto específico de negociação;
- cláusulas contratuais;
- transferências anteriores.

Além disso, jogadores de alto valor representam uma parcela pequena do dataset,
o que dificulta a aprendizagem adequada desse segmento.

---

## 14. Interpretação responsável

A saída do modelo deve ser interpretada como:

> uma estimativa estatística produzida a partir de cinco características e dos
> padrões observados no dataset utilizado.

Ela não deve ser interpretada como:

- preço real de uma futura transferência;
- avaliação oficial de um jogador;
- valor garantido de negociação;
- recomendação financeira;
- valuation profissional.

O valor de mercado e o preço efetivamente negociado em uma transferência são
conceitos diferentes.

---

## 15. Sobre confiança e incerteza

A versão atual do modelo não produz um intervalo probabilístico formal de
confiança ou de previsão.

Por isso, métricas como:

```text
50% dos erros ≤ €1,08M
80% dos erros ≤ €4,74M
90% dos erros ≤ €11,98M
```

devem ser interpretadas apenas como resultados empíricos observados no conjunto
de teste.

Elas não significam, por exemplo:

```text
"o modelo possui 90% de precisão"
```

ou:

```text
"a previsão possui 90% de confiança"
```

Uma versão futura poderá investigar métodos específicos para quantificação de
incerteza, como intervalos de previsão ou conformal prediction.

---

## 16. Uso recomendado

O modelo é adequado para:

- demonstração educacional de regressão;
- projeto de portfólio em Machine Learning;
- estudo de preparação e análise de dados;
- demonstração de avaliação de modelos;
- exploração de limitações e análise de erros;
- demonstração de um pipeline simples de Data Science.

---

## 17. Uso não recomendado

O modelo não deve ser utilizado para:

- decisões reais de compra ou venda de jogadores;
- negociação profissional de transferências;
- análise financeira de clubes;
- scouting profissional;
- decisões de investimento;
- avaliação contratual de atletas.

---

## 18. Reprodutibilidade

A divisão treino/teste utiliza:

```python
random_state=42
```

O processo pode ser reproduzido executando:

```powershell
python src\prepare_data.py
python src\train_model.py
python src\evaluate_model.py
```

O modelo treinado é salvo em:

```text
models/market_value_model.joblib
```

---

## 19. Status

Esta Model Card descreve o modelo utilizado no MVP atual do projeto.

O modelo está considerado suficiente para o objetivo educacional e de
portfólio desta versão.

Melhorias futuras devem ser tratadas como novos experimentos e não como
requisitos necessários para considerar o MVP funcional.