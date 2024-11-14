import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Carregar os dados
@st.cache_data
def load_data():
    path = 'C:/Users/italo/OneDrive/Área de Trabalho/Reconhecimento de padrões/Pred/df1.csv'
    df1 = pd.read_csv(path)
    return df1

# Função para treinar o modelo Random Forest com hiperparâmetros ajustáveis
def train_random_forest(n_estimators, max_depth, min_samples_leaf, random_state=42):
    # Selecionar as features e o target
    df = load_data()
    
    X = df.drop('Life expectation', axis=1)
    y = df['Life expectation']
    
    # Dividir os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    
    # Escalar as features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Instanciar e treinar o modelo Random Forest
    rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, 
                               min_samples_leaf=min_samples_leaf, random_state=random_state)
    rf.fit(X_train, y_train)
    
    # Fazer previsões
    y_pred = rf.predict(X_test)
    
    # Calcular métricas de desempenho
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Obter a importância das features
    feature_importances = rf.feature_importances_
    
    # Salvar o modelo e o scaler
    model_path = 'C:/Users/italo/OneDrive/Área de Trabalho/Reconhecimento de padrões/Pred/random_forest_model.pkl'
    scaler_path = 'C:/Users/italo/OneDrive/Área de Trabalho/Reconhecimento de padrões/Pred/scaler.pkl'
    joblib.dump(rf, model_path)
    joblib.dump(scaler, scaler_path)
    
    return rf, scaler, mae, mse, r2, y_test, y_pred, feature_importances, X.columns

# Função para carregar o modelo salvo e fazer previsões
def load_model_and_predict(input_data):
    model_path = 'C:/Users/italo/OneDrive/Área de Trabalho/Reconhecimento de padrões/Pred/random_forest_model.pkl'
    scaler_path = 'C:/Users/italo/OneDrive/Área de Trabalho/Reconhecimento de padrões/Pred/scaler.pkl'

    rf_model_loaded = joblib.load(model_path)
    scaler_loaded = joblib.load(scaler_path)

    input_data_scaled = scaler_loaded.transform(input_data)  # Escalar os dados de entrada
    prediction = rf_model_loaded.predict(input_data_scaled)  # Fazer a previsão
    return prediction

# Interface de ajuste de hiperparâmetros com Streamlit
st.title('Ajuste de Hiperparâmetros do Random Forest para Previsão de Expectativa de Vida')

# Ajuste dos Hiperparâmetros
st.sidebar.header('Ajuste os Hiperparâmetros do Modelo')
n_estimators = st.sidebar.slider('Número de Árvores (n_estimators)', 2, 100, 12, 1)
max_depth = st.sidebar.slider('Profundidade Máxima (max_depth)', 1, 20, 9, 1)
min_samples_leaf = st.sidebar.slider('Mínimo de Amostras por Folha (min_samples_leaf)', 1, 10, 4, 1)

# Treinar o modelo com os hiperparâmetros ajustados
if st.sidebar.button('Treinar Modelo'):
    rf, scaler, mae, mse, r2, y_test, y_pred, feature_importances, feature_names = train_random_forest(
        n_estimators, max_depth, min_samples_leaf)
    
    # Exibir métricas de desempenho
    st.write("**Mean Absolute Error (MAE):**", mae)
    st.write("**Mean Squared Error (MSE):**", mse)
    st.write("**R-squared (R2):**", r2)
    
    # Exibir importância das features
    st.write("### Importância das Features")
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    st.bar_chart(feature_importance_df.set_index('Feature'))

    # Exibir comparação de valores reais e previstos
    st.write("### Comparação de Valores Reais e Previstos")
    comparison_df = pd.DataFrame({'Real': y_test, 'Predicted': y_pred})
    st.line_chart(comparison_df)

# Seção para fazer previsões usando o modelo carregado
st.sidebar.header('Fazer Previsões')
df = load_data()
feature_names = df.drop('Life expectation', axis=1).columns

# Criar caixas de entrada para cada feature
input_data = []
for feature in feature_names:
    value = st.sidebar.number_input(f'Valor para {feature}', value=0.0)
    input_data.append(value)

# Fazer previsão com os dados inseridos
if st.sidebar.button('Prever com Dados de Entrada'):
    input_array = np.array(input_data).reshape(1, -1)
    prediction = load_model_and_predict(input_array)
    st.write("**Previsão de Expectativa de Vida:**", prediction[0])
