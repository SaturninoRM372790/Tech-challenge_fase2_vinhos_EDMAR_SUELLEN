# Classificação da Qualidade de Vinhos

**Tech Challenge — Fase 2**  
**Pós-Tech Data Analytics — FIAP**

## Autores

- Edmar Saturnino César — RM372790
- Francisca Suellen Soares Barros — RM372732

## 1. Contexto

Este projeto desenvolve um modelo de Machine Learning para classificar vinhos com base em suas características físico-químicas.

A base utilizada foi o Wine Quality Dataset, disponibilizado no arquivo `WineQT.csv`.

## 2. Objetivo

Classificar os vinhos em duas categorias:

- **Baixa ou média qualidade:** nota inferior a 7;
- **Alta qualidade:** nota igual ou superior a 7.

A variável-alvo foi criada por meio da regra:

    alta_qualidade = 1 if quality >= 7 else 0

## 3. Metodologia

O projeto foi desenvolvido com base nas etapas do processo KDD:

1. compreensão do problema;
2. seleção e inspeção dos dados;
3. limpeza e transformação;
4. análise exploratória;
5. preparação para modelagem;
6. treinamento e comparação dos algoritmos;
7. ajuste de hiperparâmetros;
8. avaliação final;
9. interpretação dos resultados.

## 4. Tratamento dos dados

A base original possuía 1.143 registros.

Após a remoção da coluna identificadora e das duplicidades, foram mantidas 1.018 amostras.

As principais verificações realizadas foram:

- análise dos tipos de dados;
- identificação de valores nulos;
- identificação e remoção de duplicidades;
- análise de outliers pelo intervalo interquartil;
- avaliação da distribuição das características;
- análise de correlações;
- análise do desbalanceamento da variável-alvo.

Não foram identificados valores nulos.

Os possíveis outliers foram mantidos, pois podem representar características legítimas dos vinhos analisados.

## 5. Desbalanceamento das classes

A variável-alvo apresentou a seguinte distribuição:

- baixa ou média qualidade: 86,54%;
- alta qualidade: 13,46%.

Para reduzir os efeitos desse desbalanceamento, foram adotadas:

- divisão estratificada dos dados;
- validação cruzada estratificada;
- ponderação automática das classes;
- utilização de recall, F1-score e ROC-AUC;
- comparação com um modelo baseline.

## 6. Modelos avaliados

Foram avaliados os seguintes algoritmos:

- Dummy Classifier;
- Regressão Logística;
- K-Nearest Neighbors;
- Árvore de Decisão;
- Random Forest.

A Regressão Logística e a Árvore de Decisão apresentaram os resultados mais promissores e foram submetidas ao ajuste de hiperparâmetros com `GridSearchCV`.

O F1-score foi adotado como principal critério de seleção.

## 7. Modelo final

O modelo selecionado foi a **Regressão Logística ajustada**.

O pipeline final inclui:

- padronização das características com `StandardScaler`;
- compensação do desbalanceamento com `class_weight="balanced"`;
- hiperparâmetros selecionados por validação cruzada.

## 8. Resultados

### Validação cruzada

| Métrica | Resultado |
|---|---:|
| Acurácia | 79,73% |
| Precisão | 38,57% |
| Recall | 82,73% |
| F1-score | 52,49% |
| ROC-AUC | 87,11% |

### Conjunto de teste

| Métrica | Resultado |
|---|---:|
| Acurácia | 78,92% |
| Precisão | 36,21% |
| Recall | 77,78% |
| F1-score | 49,41% |
| ROC-AUC | 88,11% |

### Matriz de confusão

| Resultado | Quantidade |
|---|---:|
| Verdadeiros negativos | 140 |
| Falsos positivos | 37 |
| Falsos negativos | 6 |
| Verdadeiros positivos | 21 |

O modelo identificou corretamente 21 dos 27 vinhos de alta qualidade presentes no conjunto de teste.

A maior diferença entre a validação cruzada e o teste foi de 4,95 pontos percentuais no recall, indicando estabilidade e ausência de sinais relevantes de overfitting.

## 9. Interpretação das características

As principais associações positivas com a classificação de alta qualidade foram:

| Característica | Coeficiente | Razão de chances |
|---|---:|---:|
| Teor alcoólico | 0,8477 | 2,3342 |
| Sulfatos | 0,7347 | 2,0848 |
| Acidez fixa | 0,6649 | 1,9444 |

As principais associações negativas foram:

| Característica | Coeficiente | Razão de chances |
|---|---:|---:|
| Densidade | -0,7671 | 0,4644 |
| Dióxido de enxofre total | -0,5852 | 0,5570 |
| Acidez volátil | -0,5762 | 0,5621 |

Os resultados representam associações preditivas e não comprovam relações causais.

## 10. Estrutura do repositório

    Tech-challenge_fase2_vinhos_EDMAR_SUELLEN/
    ├── data/
    │   └── WineQT.csv
    ├── models/
    │   └── regressao_logistica_vinhos.joblib
    ├── notebooks/
    │   └── TC_Fase2_Vitivinicola_EDMAR_SUELLEN_FIAP.ipynb
    results/
    ├── coeficientes_modelo.png
    ├── coeficientes_modelo_final.csv
    ├── comparacao_modelos.png
    ├── comparacao_validacao_teste.csv
    ├── curva_roc.png
    ├── distribuicao_classes.png
    ├── matriz_confusao.png
    ├── matriz_correlacao.png
    ├── metricas_modelo_final.csv
    ├── previsoes_conjunto_teste.csv
    ├── ranking_arvore_decisao.csv
    ├── ranking_regressao_logistica.csv
    └── resumo_projeto.json    
    ├── src/
    │   └── prever_qualidade.py
    ├── .gitignore
    ├── README.md
    └── requirements.txt

## 11. Instalação

Clone o repositório:

    git clone https://github.com/SaturninoRM372790/Tech-challenge_fase2_vinhos_EDMAR_SUELLEN.git
    cd Tech-challenge_fase2_vinhos_EDMAR_SUELLEN

Crie um ambiente virtual:

    python -m venv .venv

Ative o ambiente virtual no Windows:

    .venv\Scripts\activate

Ative o ambiente virtual no Linux ou macOS:

    source .venv/bin/activate

Instale as dependências do projeto:

    pip install -r requirements.txt

## 12. Execução de novas previsões

O arquivo `src/prever_qualidade.py` permite classificar novas amostras por linha de comando.

Exemplo:

    python src/prever_qualidade.py --entrada data/WineQT.csv

Também é possível definir o arquivo de saída:

    python src/prever_qualidade.py --entrada data/WineQT.csv --saida results/previsoes_novas_amostras.csv

O limiar padrão de classificação é 0,50.

## 13. Limitações

Entre as principais limitações do estudo estão:

- tamanho relativamente limitado da base;
- forte desbalanceamento das classes;
- apenas 27 vinhos de alta qualidade no conjunto de teste;
- subjetividade da avaliação de qualidade;
- ausência de informações sobre safra, região, variedade da uva e processo produtivo;
- precisão reduzida para a classe de alta qualidade;
- necessidade de validação em novas amostras antes de uso produtivo.

## 14. Conclusão

A Regressão Logística ajustada apresentou boa capacidade de identificar vinhos de alta qualidade e manteve resultados próximos entre validação cruzada e teste.

O recall de 77,78% demonstra que o modelo reconheceu a maior parte dos vinhos de alta qualidade presentes no teste.

Entretanto, a precisão de 36,21% indica uma quantidade relevante de falsos positivos. Por isso, o modelo deve ser utilizado como ferramenta de apoio à triagem e ao controle de qualidade, e não como substituto da avaliação sensorial ou do conhecimento enológico.
