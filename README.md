# Tech Challenge - Previsão de Preços de Ações GOOG

Este projeto foi desenvolvido como parte do Tech Challenge do curso de Pós-graduação em Machine Learning Engineering. O objetivo é criar um modelo de Machine Learning para prever os preços de ações, integrando coleta de dados, armazenamento em banco de dados e visualização dos resultados. [cite: 3, 4, 5, 6, 7, 11, 12, 13]

## Desafio do Tech Challenge

O Tech Challenge proposto envolve a construção de uma solução completa de Machine Learning, desde a coleta de dados até a apresentação dos resultados em uma aplicação funcional. Os requisitos principais do desafio são: [cite: 6, 7, 11]

* **Coleta e Armazenamento de Dados:** Desenvolver uma API para coletar dados de ações (idealmente em tempo real) e armazená-los em um banco de dados. [cite: 6]
* **Modelo de Machine Learning:** Implementar um modelo de ML para prever os preços das ações, utilizando os dados coletados. [cite: 7]
* **Documentação:** Fornecer documentação clara do código e do modelo. [cite: 8, 9, 10]
* **Apresentação Visual:** Criar uma forma de apresentar o modelo, como um dashboard. [cite: 11]

## Solução Implementada

Este projeto aborda os requisitos do Tech Challenge através das seguintes etapas:

1.  **Coleta de Dados:** Os dados históricos e em tempo real das ações da GOOG são coletados utilizando a biblioteca `yfinance`. [cite: main.py]
2.  **Armazenamento de Dados:** Os dados coletados são armazenados em um banco de dados PostgreSQL usando SQLAlchemy. [cite: main.py]
3.  **Modelo de Machine Learning:** Um modelo de Random Forest Regressor é treinado para prever os preços de fechamento das ações. [cite: main.py]
4.  **API Flask:** Uma API Flask é desenvolvida para disponibilizar os dados históricos e as previsões do modelo. [cite: main.py]
5.  **Dashboard:** Um dashboard web é construído com Flask e HTML para visualizar os dados históricos e as previsões de preços das ações. [cite: main.py, templates/dashboard.html]

## Modelo de Machine Learning

O modelo de Machine Learning utilizado é o **Random Forest Regressor**. Este modelo foi escolhido devido à sua capacidade de lidar com dados não lineares e sua robustez contra overfitting.

* **Hiperparâmetros:** O modelo é treinado com otimização de hiperparâmetros utilizando `GridSearchCV` para encontrar a melhor combinação de parâmetros. [cite: main.py]
* **Features:** O modelo utiliza como features o preço de fechamento, volume, média móvel e retorno diário da ação. [cite: main.py]
* **Avaliação:** O desempenho do modelo é avaliado utilizando as métricas de Mean Squared Error (MSE) e Mean Absolute Error (MAE). [cite: main.py]

## Como Executar o Projeto

Para executar o projeto, siga as instruções abaixo:

1.  **Pré-requisitos:**
    * Python 3.x
    * Pip
    * PostgreSQL
2.  **Instalar as dependências:**

    ```bash
    pip install -r requirements.txt
    ```
3.  **Configurar o banco de dados:**
    * Crie um banco de dados PostgreSQL.
    * Configure as variáveis de ambiente no arquivo `.env` com as credenciais do banco de dados.
4.  **Executar a API:**

    ```bash
    python main.py
    ```
    A API estará disponível em `http://localhost:5000`.
5.  **Acessar o Dashboard:**
    * Abra o navegador e acesse `http://localhost:5000/` para visualizar o dashboard.

## Rotas da API

* `/historico`: Retorna os dados históricos das ações.
* `/prever`: Retorna a previsão do preço da ação para o dia seguinte.
* `/treinarmodelo`: Coleta os dados e treina o modelo (deve ser executado para treinar o modelo antes de realizar a previsão).
* `/`: Retorna o dashboard

---

Este README fornece uma visão geral do projeto, explicando seu propósito no contexto do Tech Challenge, descrevendo o modelo de Machine Learning utilizado e fornecendo instruções sobre como executar o projeto.
