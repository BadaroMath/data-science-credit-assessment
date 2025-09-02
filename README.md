# Credit Risk Modeling & Data Pipeline Case Study

Este repositório contém a resolução de um case técnico para uma posição de Especialista em Dados. O projeto demonstra um fluxo de trabalho completo, desde a ingestão e processamento de dados semi-estruturados (XML) até a construção e avaliação de um modelo de classificação de risco de crédito.

## Objetivos do Projeto

O desafio foi dividido em duas partes principais:

1.  **Engenharia de Dados (Análise de Financiamentos)**: Desenvolver um pipeline para processar dados de histórico de crédito em formato XML, armazená-los em um formato analítico e extrair informações via SQL.
2.  **Modelagem Estatística (Análise de Risco)**: Construir um modelo preditivo para discriminar clientes com base em um conjunto de variáveis anonimizadas, seguindo um ciclo completo de EDA, seleção de features e comparação de algoritmos.

## Estrutura do Repositório

```
/
├── Anexo 2 - Base Modelagem.csv   # Base de dados para a modelagem
├── main.ipynb                     # Notebook principal com toda a resolução e análise
├── data_class.py                  # Módulo com as dataclasses para o parsing do XML
├── data_parser.py                 # Módulo com o parser do arquivo XML
├── bigquery_uploader.py           # Módulo auxiliar para upload no Google BigQuery
├── example.py                     # Módulo que gera um XML de exemplo para testes
└── README.md                      # Este arquivo
```

## Destaques Técnicos

### Parte 1: Engenharia de Dados

-   **Parsing de XML**: Implementação de um parser robusto e orientado a objetos em Python para extrair dados de uma estrutura XML complexa.
-   **Modelagem de Dados Analíticos**:
    -   Proposta de armazenamento em **Google BigQuery** utilizando um schema desnormalizado com campos aninhados para otimizar consultas analíticas.
    -   Apresentação de uma **arquitetura alternativa** baseada em um modelo relacional clássico (Esquema Estrela) para bancos como SQL Server ou PostgreSQL.
-   **Consultas SQL**: Demonstração de consultas complexas (`UNNEST`, `JOINs`, funções de janela) para extrair valor dos dados em ambas as arquiteturas.

### Parte 2: Modelagem Preditiva

-   **Análise Exploratória de Dados (EDA)**: Investigação completa dos dados, incluindo tratamento de valores ausentes e análise de desbalanceamento de classes.
-   **Seleção de Features**: Utilização de uma metodologia sistemática com **Recursive Feature Elimination (RFE)** para selecionar as 50 variáveis mais preditivas.
-   **Comparação de Modelos**: Avaliação e comparação de múltiplos algoritmos de classificação (Logistic Regression, Random Forest, XGBoost) através de validação cruzada para garantir a robustez da escolha.
-   **Otimização de Hiperparâmetros**: Tuning fino do melhor modelo (`Logistic Ridge`) usando `RandomizedSearchCV` para maximizar a performance.
-   **Interpretação de Resultados**: Análise detalhada das métricas finais (AUC-ROC, Precision, Recall) e discussão sobre o trade-off e a aplicabilidade do modelo no contexto de negócio.

## Resultados Finais

O modelo final, um **Logistic Ridge** otimizado, alcançou uma **AUC-ROC de 0.6635** no conjunto de teste. O modelo conseguiu identificar **57%** dos casos de maior risco (recall), posicionando-se como uma ferramenta valiosa para um sistema de alerta precoce.


---