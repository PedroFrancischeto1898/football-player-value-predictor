# Development — Football Player Value Predictor

## 1. Objetivo deste documento

Este documento registra o processo de desenvolvimento do projeto
**Football Player Value Predictor**.

O objetivo não é apenas apresentar o resultado final, mas documentar:

- o problema inicial;
- as decisões tomadas;
- as alternativas consideradas;
- os problemas encontrados;
- os experimentos realizados;
- os resultados obtidos;
- as limitações identificadas;
- a evolução do projeto até o MVP.

O projeto foi desenvolvido como um projeto educacional e de portfólio, com
ênfase em simplicidade, compreensão do processo e capacidade de explicar as
decisões técnicas.

---

# 2. Definição do problema

A ideia inicial foi construir um projeto simples de Machine Learning aplicado
ao futebol.

A pergunta escolhida foi:

> É possível estimar o valor de mercado de um jogador utilizando informações
> básicas como idade, posição, minutos jogados, gols e assistências?

O objetivo não era reproduzir sistemas profissionais de avaliação de atletas.

O projeto deveria demonstrar um pipeline simples de Data Science:

```text
dados
  ↓
preparação
  ↓
modelo
  ↓
avaliação
  ↓
previsão
```

---

# 3. Definição do MVP

O escopo do MVP foi deliberadamente reduzido.

Foram escolhidas apenas cinco características:

```text
age
position
minutes
goals
assists
```

Target:

```text
market_value_in_eur
```

Modelo inicialmente escolhido:

```python
LinearRegression()
```

Essa escolha foi feita para manter o modelo:

- simples;
- interpretável;
- fácil de explicar;
- adequado ao objetivo educacional.

Não foram incluídos inicialmente:

- modelos complexos;
- dezenas de features;
- informações financeiras de clubes;
- dados de compradores e vendedores;
- tuning de hiperparâmetros;
- redes neurais;
- pipelines sofisticados.

A simplicidade foi tratada como um requisito do projeto.

---

# 4. Estrutura inicial do projeto

Foi criada uma estrutura pequena:

```text
football-player-value-predictor/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── reports/
│   └── figures/
│
├── src/
│
├── .gitignore
├── README.md
└── requirements.txt
```

O objetivo era criar apenas as pastas realmente necessárias.

Posteriormente foram adicionados:

```text
models/
PROJECT_CONTEXT.md
MODEL_CARD.md
DEVELOPMENT.md
```

---

# 5. Ambiente de desenvolvimento

O projeto foi desenvolvido em:

- Windows;
- PowerShell;
- Python 3.11;
- Visual Studio Code.

Foi criado um ambiente virtual:

```powershell
py -3.11 -m venv .venv
```

As principais dependências utilizadas foram:

```text
pandas
numpy
scikit-learn
matplotlib
joblib
requests
```

O ambiente virtual foi mantido fora do controle de versão utilizando
`.gitignore`.

---

# 6. Fonte de dados

Foi escolhido o dataset público:

```text
dcaribou/transfermarkt-datasets
```

Foram utilizados dois arquivos:

```text
players.csv.gz
appearances.csv.gz
```

A escolha permitiu obter:

- informações dos jogadores;
- valores de mercado;
- datas de nascimento;
- posições;
- minutos;
- gols;
- assistências.

---

# 7. Preparação dos dados

Foi criado:

```text
src/prepare_data.py
```

O script realiza:

1. leitura dos dados de jogadores;
2. leitura das aparições;
3. conversão das datas;
4. filtragem do período;
5. agregação das estatísticas por jogador;
6. cálculo da idade;
7. combinação das tabelas;
8. remoção de dados inválidos;
9. geração do dataset final.

Período utilizado:

```text
01/07/2025 a 12/06/2026
```

Dataset final:

```text
data/processed/players_model.csv
```

Resultado:

```text
8.187 jogadores
```

---

# 8. Primeiro modelo

Foi criado:

```text
src/train_model.py
```

O primeiro modelo utilizou:

```python
LinearRegression()
```

com o valor de mercado diretamente em euros como target.

As posições foram transformadas com:

```python
OneHotEncoder(handle_unknown="ignore")
```

A divisão dos dados foi:

```text
80% treino
20% teste
random_state = 42
```

Resultado:

```text
Treino: 6.549 jogadores
Teste: 1.638 jogadores

MAE: €5.648.020
R²: 0,3462
```

---

# 9. Primeiro problema identificado

Foi gerado um gráfico de:

```text
valor real × valor previsto
```

A inspeção visual mostrou três comportamentos importantes:

1. grande concentração de jogadores de baixo valor;
2. forte subestimação de jogadores muito caros;
3. existência de previsões negativas.

O terceiro comportamento era matematicamente possível em uma regressão linear,
mas não fazia sentido para um valor de mercado.

Além disso, a distribuição dos valores era altamente assimétrica.

---

# 10. Hipótese: transformar o target

A partir da análise do gráfico, foi testada uma única alteração:

```python
np.log1p(market_value_in_eur)
```

A ideia era reduzir o impacto dos valores extremos durante o treinamento.

Após a previsão, os valores foram convertidos novamente para euros:

```python
np.expm1(prediction)
```

As previsões negativas também foram limitadas a zero.

Nenhuma feature foi alterada.

O algoritmo permaneceu:

```python
LinearRegression()
```

---

# 11. Resultado do segundo experimento

O modelo com target logarítmico apresentou:

```text
MAE: €4.276.655
R² em euros: 0,2867
```

Comparação:

| Modelo | MAE | R² |
|---|---:|---:|
| Target direto em euros | €5.648.020 | 0,3462 |
| Target logarítmico | €4.276.655 | 0,2867 |

O MAE caiu aproximadamente 24%.

O R² em euros ficou menor.

Isso mostrou que diferentes métricas podem contar histórias diferentes.

A decisão foi manter o modelo logarítmico porque ele apresentou menor erro
absoluto médio para o conjunto de teste e comportamento mais adequado para a
maior parte dos jogadores.

A limitação nos jogadores de alto valor foi mantida documentada.

---

# 12. Persistência do modelo

Para evitar a necessidade de treinar o modelo a cada previsão, foi adicionada
persistência com:

```python
joblib
```

O modelo passou a ser salvo em:

```text
models/market_value_model.joblib
```

Isso permitiu separar:

```text
treinamento
```

de:

```text
uso do modelo
```

---

# 13. Construção do preditor

Foi criado:

```text
src/predict.py
```

O programa recebe:

```text
idade
posição
minutos
gols
assistências
```

e retorna:

```text
valor de mercado estimado
```

Exemplo utilizado durante o desenvolvimento:

```text
Idade: 21
Posição: Attack
Minutos: 3000
Gols: 18
Assistências: 10
```

Resultado:

```text
€46.859.871
```

Durante o teste foi identificado um pequeno problema de usabilidade:

```text
attack
```

não era aceito, enquanto:

```text
Attack
```

era aceito.

A entrada foi normalizada com:

```python
.title()
```

permitindo o uso de maiúsculas ou minúsculas.

Esse ajuste foi feito na interface de entrada, sem alterar o modelo.

---

# 14. Controle de versão

Depois que o MVP básico estava funcionando, o projeto foi versionado com Git.

Foram utilizados:

```text
git init
git add
git commit
git remote
git push
```

O projeto foi publicado no GitHub.

Arquivos grandes de dados foram mantidos fora do repositório por meio do
`.gitignore`.

O modelo treinado foi mantido no repositório por ser um arquivo pequeno e
permitir a execução direta do preditor.

---

# 15. Documentação inicial

Foram criados:

```text
README.md
PROJECT_CONTEXT.md
```

O `README.md` apresenta o projeto publicamente.

O `PROJECT_CONTEXT.md` registra:

- escopo;
- decisões;
- estado atual;
- restrições;
- próximos passos.

Durante a publicação no GitHub, foi identificado que o README estava
incompleto.

O problema foi corrigido e uma versão completa passou a ser utilizada.

---

# 16. Avaliação ampliada

Depois do MVP funcional, foi criada uma etapa específica para avaliar a
qualidade do modelo.

Foi criado:

```text
src/evaluate_model.py
```

O objetivo não foi melhorar o modelo, mas entender melhor seu desempenho.

Foram avaliados:

- baseline;
- MAE;
- R²;
- erro mediano;
- percentis dos erros;
- desempenho por faixa de valor.

---

# 17. Baseline

Foi criado um baseline simples:

> prever para qualquer jogador a mediana do valor de mercado do conjunto de
> treinamento.

Valor do baseline:

```text
€1.500.000
```

Resultado:

```text
MAE baseline: €5.195.650
MAE modelo:   €4.276.655
```

Melhoria:

```text
17,7%
```

Esse resultado mostrou que o modelo aprende informação útil a partir das
features utilizadas e supera uma previsão constante simples.

---

# 18. Distribuição dos erros

Foi analisada a distribuição dos erros absolutos.

Resultado:

```text
50% das previsões:
erro de até €1.082.526

80% das previsões:
erro de até €4.735.271

90% das previsões:
erro de até €11.976.072
```

Essa análise tornou a avaliação mais intuitiva do que observar apenas o MAE.

Também ficou registrado que esses valores não representam intervalos formais
de confiança.

---

# 19. Avaliação por faixa de valor

A análise por faixa revelou uma limitação importante.

| Valor real | Jogadores | MAE |
|---|---:|---:|
| < €5M | 1.202 | €1.146.089 |
| €5M–€20M | 296 | €6.161.646 |
| €20M–€50M | 109 | €20.762.384 |
| ≥ €50M | 31 | €49.697.238 |

O modelo apresenta desempenho muito melhor para jogadores de menor valor.

Jogadores de alto valor são significativamente mais difíceis de estimar com as
cinco features utilizadas.

Essa análise também demonstrou que uma única métrica global pode esconder
diferenças importantes entre segmentos.

---

# 20. Model Card

Foi criado:

```text
MODEL_CARD.md
```

O documento registra:

- objetivo do modelo;
- dados;
- features;
- algoritmo;
- transformação do target;
- baseline;
- métricas;
- distribuição dos erros;
- desempenho por faixa;
- limitações;
- uso recomendado;
- uso não recomendado.

A Model Card serve como documentação técnica e como forma de comunicação
responsável das capacidades do modelo.

---

# 21. Princípios adotados durante o desenvolvimento

## Simplicidade

O projeto não deveria crescer apenas porque novas técnicas estavam disponíveis.

Uma nova funcionalidade só deveria ser adicionada quando existisse uma razão
clara.

## Uma alteração por vez

Os experimentos foram realizados de forma incremental.

Exemplo:

```text
baseline em euros
        ↓
análise do gráfico
        ↓
hipótese
        ↓
target log
        ↓
nova avaliação
```

Isso tornou possível entender o efeito de cada decisão.

## Avaliação antes de complexidade

Antes de adicionar novos algoritmos, o foco foi entender o modelo existente.

## Transparência

As limitações foram documentadas junto com os resultados positivos.

---

# 22. O que foi deliberadamente deixado de fora

O MVP não inclui:

- Random Forest;
- XGBoost;
- redes neurais;
- tuning de hiperparâmetros;
- informações financeiras dos clubes;
- comprador e vendedor;
- duração contratual;
- reputação;
- potencial;
- força da liga;
- dezenas de features adicionais.

Esses itens poderiam ser investigados futuramente, mas não eram necessários
para demonstrar o ciclo completo do projeto.

---

# 23. Competências demonstradas

O desenvolvimento deste projeto envolve:

### Python

- scripts;
- funções;
- módulos;
- ambiente virtual;
- dependências.

### Manipulação de dados

- pandas;
- filtros;
- joins;
- agregações;
- tratamento de datas.

### Machine Learning

- regressão;
- encoding de variável categórica;
- transformação de target;
- treino/teste;
- MAE;
- R²;
- baseline;
- análise de erro.

### Engenharia de software básica

- organização de diretórios;
- separação de responsabilidades;
- persistência de modelo;
- controle de versão;
- documentação;
- reprodutibilidade.

### Git e GitHub

- commits;
- branches;
- remote;
- merge;
- resolução de conflito;
- push;
- repositório público.

---

# 24. Estado atual do MVP

Atualmente o projeto permite:

```text
obter dados
    ↓
preparar dataset
    ↓
treinar modelo
    ↓
avaliar modelo
    ↓
salvar modelo
    ↓
carregar modelo
    ↓
informar características
    ↓
obter uma previsão
```

Além disso, o projeto possui documentação sobre:

- funcionamento;
- contexto;
- desempenho;
- limitações;
- processo de desenvolvimento.

---

# 25. Próximas etapas

As próximas etapas planejadas para o MVP são:

```text
interface Streamlit
        ↓
publicação da interface
        ↓
teste de reprodutibilidade
        ↓
revisão final da documentação
        ↓
encerramento do MVP
```

Uma eventual melhoria do algoritmo deverá ser tratada como uma versão futura e
não como requisito para concluir este MVP.

---

# 26. Principal aprendizado

O principal objetivo deste projeto não foi obter a maior métrica possível.

O foco foi construir e compreender um ciclo completo:

```text
problema
   ↓
dados
   ↓
implementação
   ↓
avaliação
   ↓
identificação de problema
   ↓
experimento
   ↓
decisão
   ↓
documentação
   ↓
software utilizável
```

Esse processo é o principal resultado técnico e educacional do projeto.