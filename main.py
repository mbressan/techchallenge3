# Importações
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from flask import Flask, request, jsonify, render_template
from sqlalchemy import create_engine, Column, Integer, Float, Date, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os
import base64
from io import BytesIO
from datetime import datetime, timedelta
import pickle




# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# Configuração do SQLAlchemy
Base = declarative_base()
DATABASE_URL = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}/{os.getenv('DB_NAME')}"
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)

# Modelo da tabela
class HistoricoAcao(Base):
    __tablename__ = 'historico_acao'
    id = Column(Integer, primary_key=True, autoincrement=True)
    data_pregao = Column(Date, nullable=False)
    abertura = Column(Float)
    alta = Column(Float)
    baixa = Column(Float)
    fechamento = Column(Float)
    volume = Column(Integer)

# Criar tabelas no banco de dados
Base.metadata.create_all(engine)

# Classe para Coleta de Dados
class ColetorDeDados:
    """Classe para coleta de dados históricos de ações."""

    def __init__(self, ticker, data_inicio, data_fim):
        self.ticker = ticker
        self.data_inicio = data_inicio
        self.data_fim = data_fim

    def coletar_dados_historicos(self):
        """Coleta dados históricos de ações usando a biblioteca yfinance."""
        dados = yf.download(self.ticker, start=self.data_inicio, end=self.data_fim)
        dados.reset_index(inplace=True)
        return dados

    def coletar_dados_tempo_real(self):
        """Coleta dados em tempo real de ações usando a biblioteca yfinance."""
        dados = yf.download(self.ticker, period='1d', interval='1m')
        if isinstance(dados.columns, pd.MultiIndex):
            dados = dados.droplevel(1, axis=1)
        dados.rename(columns={'Open': 'abertura', 'High': 'alta', 'Low': 'baixa', 'Close': 'fechamento', 'Volume': 'volume'}, inplace=True)
        dados = dados[['abertura', 'alta', 'baixa', 'fechamento', 'volume']]
        dados.reset_index(inplace=True)  # Garante que o índice seja convertido em uma coluna
        dados.rename(columns={'Datetime': 'data'}, inplace=True)  # Renomeia a coluna de índice para 'data'
        return dados

# Funções de Banco de Dados (SQLAlchemy)

def armazenar_dados_rds(dados, nome_tabela):
    """Armazena os dados no banco de dados usando SQLAlchemy."""

    #exlui os dados antigos
    excluir_dados_rds(nome_tabela)

    session = Session()
    try:
        for _, row in dados.iterrows():
            registro = HistoricoAcao(
                data_pregao=row.iloc[0], 
                abertura=row.iloc[1],    
                alta=row.iloc[2],        
                baixa=row.iloc[3],       
                fechamento=row.iloc[4],  
                volume=row.iloc[5]       
            )
            session.add(registro)
        session.commit()
    except Exception as e:
        print(f"Erro ao armazenar dados no RDS: {e}")
        session.rollback()
    finally:
        session.close()


def excluir_dados_rds(nome_tabela):
    """Exclui os dados do banco de dados usando SQLAlchemy."""
    session = Session()
    try:
        session.query(HistoricoAcao).delete()
        session.commit()
    except Exception as e:
        print(f"Erro ao excluir dados do RDS: {e}")
        session.rollback()
    finally:
        session.close()


def ler_dados_rds(nome_tabela):
    """Lê os dados do banco de dados usando SQLAlchemy."""
    session = Session()
    try:
        dados = session.query(HistoricoAcao).all()
        return pd.DataFrame([{
            'data_pregao': registro.data_pregao,
            'abertura': registro.abertura,
            'alta': registro.alta,
            'baixa': registro.baixa,
            'fechamento': registro.fechamento,
            'volume': registro.volume
        } for registro in dados])
    except Exception as e:
        print(f"Erro ao ler dados do RDS: {e}")
    finally:
        session.close()


# Funções de Pré-processamento
def explorar_dados(dados):
    """Explora os dados com estatísticas descritivas e visualizações."""
    print(dados.describe())
    plt.figure(figsize=(14, 7))
    plt.plot(dados['data_pregao'], dados['fechamento'], label='Preço de Fechamento')
    plt.xlabel('Data')
    plt.ylabel('Preço')
    plt.title('Preço de Fechamento ao Longo do Tempo')
    plt.legend()
    plt.show()


def pre_processar_dados(dados):
    """Pré-processa os dados para modelagem."""
    dados['media_movel'] = dados['fechamento'].rolling(window=7).mean()
    dados['retorno_diario'] = dados['fechamento'].pct_change()
    dados.dropna(inplace=True)

    scaler = MinMaxScaler()
    dados[['fechamento', 'volume', 'media_movel', 'retorno_diario']] = scaler.fit_transform(
        dados[['fechamento', 'volume', 'media_movel', 'retorno_diario']]
    )

    X = dados[['fechamento', 'volume', 'media_movel', 'retorno_diario']]
    y = dados['fechamento']
    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, shuffle=False)
    return X_treino, X_teste, y_treino, y_teste, scaler


# Funções de Modelagem
def treinar_modelo(X_treino, y_treino):
    """Treina o modelo Random Forest Regressor com otimização de hiperparâmetros."""
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    modelo = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(modelo, param_grid, cv=3, scoring='neg_mean_squared_error')
    grid_search.fit(X_treino, y_treino)
    print(f"Melhores hiperparâmetros: {grid_search.best_params_}")
    return grid_search.best_estimator_


def avaliar_modelo(modelo, X_teste, y_teste):
    """Avalia o modelo e exibe métricas e gráficos."""
    previsoes = modelo.predict(X_teste)
    mse = mean_squared_error(y_teste, previsoes)
    mae = mean_absolute_error(y_teste, previsoes)
    print(f"MSE: {mse}, MAE: {mae}")

def gerar_grafico_historico(previsao=None):
    """Gera o gráfico de histórico de preços com a previsão de amanhã."""
    dados = ler_dados_rds("goog_historico")
    if dados is None or dados.empty:
        return None  # Retorna None em caso de erro ou dados vazios

    plt.figure(figsize=(10, 5))
    plt.plot(dados['data_pregao'], dados['fechamento'], label='Preço de Fechamento', color='blue')
    
    # Adicionar o ponto da previsão de amanhã, se disponível
    if previsao:
        plt.scatter(datetime.now() + timedelta(days=1), previsao, color='red', label='Previsão de Amanhã')
    
    plt.xlabel('Data')
    plt.ylabel('Preço (USD)')
    plt.title('Histórico de Preços de GOOG com Previsão')
    plt.legend()
    plt.grid(True)

    # Salvar o gráfico em um buffer na memória
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_url = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()

    return plot_url

def gerar_grafico_previsao():
    """Obtém a previsão do preço de amanhã."""
    ticker = "GOOG"
    hoje = datetime.now().strftime("%Y-%m-%d")
    coletor = ColetorDeDados(ticker, hoje, hoje)
    dados_recentes = coletor.coletar_dados_tempo_real()

    if dados_recentes.empty:
        return None, "Não há dados recentes para gerar a previsão."

    previsao_response, status_code = prever_acao()

    if status_code != 200:
        return None, f"Erro ao obter a previsão: {previsao_response.get_json()['mensagem']}"

    previsao = previsao_response.get_json()['previsao']
    return None, f"Previsão para amanhã: ${previsao:.2f}"

# API Flask

app = Flask(__name__)

# Importa as rotas

@app.route('/historico', methods=['GET'])
def obter_historico():
    """Retorna os dados históricos das ações."""
    dados = ler_dados_rds("goog_historico")
    if dados is not None:
        return jsonify(dados.to_dict(orient='records')), 200
    return jsonify({"mensagem": "Erro ao obter dados históricos"}), 500

@app.route('/prever', methods=['GET'])
def prever_acao():
    """Realiza a previsão do valor da ação com base nos dados fornecidos."""
    
    # Coletar os dados da ação em tempo real
    ticker = "GOOG"
    coletor = ColetorDeDados(ticker, datetime.now().strftime("%Y-%m-%d"), datetime.now().strftime("%Y-%m-%d"))
    dados_reais = coletor.coletar_dados_tempo_real()

    # Validar os dados de entrada
    if dados_reais.empty:
        return jsonify({"mensagem": "Não foram encontrados dados diarios da Ação"}), 400


    dados_reais['media_movel'] = dados_reais['fechamento'].rolling(window=20).mean()
    dados_reais['retorno_diario'] = dados_reais['fechamento'].pct_change()
    dados_reais.dropna(inplace=True)
  
    # # Verifica se o modelo e o scaler estão carregados
    with open('modelo.pkl', 'rb') as arquivo_modelo:
        modelo = pickle.load(arquivo_modelo)
    with open('scaler.pkl', 'rb') as arquivo_scaler:
        scaler = pickle.load(arquivo_scaler)

    # Certifique-se de que a ordem e os nomes das chaves correspondem às features do modelo
    features = ['fechamento', 'volume', 'media_movel', 'retorno_diario']

    # Extrair os últimos valores das features
    input_data = [dados_reais[feature].iloc[-1] for feature in features]

    # Verificar se algum valor está ausente
    if any(pd.isna(value) for value in input_data):
        return jsonify({"mensagem": "Dados de entrada incompletos"}), 400
    
    # Transformar os dados de entrada usando o mesmo scaler usado no treinamento
    input_scaled = scaler.transform([input_data])

    # Realizar a previsão
    previsao = modelo.predict(input_scaled)

    # Inverter a escala da previsão para a escala original
    previsao_original = scaler.inverse_transform(input_scaled)[:, 0][0]

    return jsonify({"previsao": previsao_original}), 200


@app.route('/treinarmodelo', methods=['GET'])
def coletar_dados_treinar():
    print("Iniciando o treinamento do modelo...")
    
    # Coletar e processar dados históricos até o dia anterior
    ticker = "GOOG"
    data_inicio = "2022-01-01"
    data_fim = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    print(f"Coletando dados de {ticker} de {data_inicio} a {data_fim}...")

    coletor = ColetorDeDados(ticker, data_inicio, data_fim)
    dados = coletor.coletar_dados_historicos()
    
    armazenar_dados_rds(dados, "goog_historico")
    dados = ler_dados_rds("goog_historico")
    
    # Explorar e pré-processar dados
    #explorar_dados(dados)
    X_treino, X_teste, y_treino, y_teste, scaler = pre_processar_dados(dados)
    
    # Treinar e avaliar o modelo
    modelo = treinar_modelo(X_treino, y_treino)
    avaliar_modelo(modelo, X_teste, y_teste)

    # Salvar o modelo treinado  e scaler
    with open('modelo.pkl', 'wb') as arquivo_modelo:
        pickle.dump(modelo, arquivo_modelo)
    with open('scaler.pkl', 'wb') as arquivo_scaler:
        pickle.dump(scaler, arquivo_scaler)

    return jsonify("Dados coletados e modelo treinado!"), 200


@app.route("/")
def dashboard():
    """Rota principal do dashboard."""
    grafico_previsao, mensagem_previsao = gerar_grafico_previsao()
    grafico_historico = gerar_grafico_historico(previsao=float(mensagem_previsao.split('$')[1]))

    return render_template(
        "dashboard.html",
        grafico_historico=grafico_historico,
        mensagem_previsao=mensagem_previsao
    )

# Exemplo de Uso (para teste)
if __name__ == '__main__':

    # Iniciar a API
    app.run(debug=False, use_reloader=False)