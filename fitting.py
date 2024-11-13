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
    model_path = 'C:/Users/italo/OneDrive/Área de Trabalho/Reconhecimento de padrões/Pred/random_forest_model_5.pkl'
    scaler_path = 'C:/Users/italo/OneDrive/Área de Trabalho/Reconhecimento de padrões/Pred/scaler.pkl'
    joblib.dump(rf, model_path)
    joblib.dump(scaler, scaler_path)
    
    return rf, scaler, mae, mse, r2, y_test, y_pred, feature_importances, X.columns

# Função para carregar o modelo salvo e fazer previsões
def load_model_and_predict(input_data):
    model_path = r'C:/Users/italo/OneDrive/Área de Trabalho/Reconhecimento de padrões/Pred/random_forest_model_5.pkl'
    scaler_path = r'C:/Users/italo/OneDrive/Área de Trabalho/Reconhecimento de padrões/Pred/scaler.pkl'

    rf_model_loaded = joblib.load(model_path)
    scaler_loaded = joblib.load(scaler_path)

    input_data_scaled = scaler_loaded.transform(input_data)  # Escalar os dados de entrada
    prediction = rf_model_loaded.predict(input_data_scaled)  # Fazer a previsão
    return prediction

# Interface de ajuste de hiperparâmetros com Streamlit
st.title('Ajuste de Hiperparâmetros do Random Forest e Índice de saúde do transformador')

# Ajuste dos Hiperparâmetros
st.sidebar.header('Ajuste os Hiperparâmetros do Modelo')
n_estimators = st.sidebar.slider('Número de Árvores (n_estimators)', 2, 100, 12, 1)
max_depth = st.sidebar.slider('Profundidade Máxima (max_depth)', 1, 20, 9, 1)
min_samples_leaf = st.sidebar.slider('Mínimo de Amostras por Folha (min_samples_leaf)', 1, 10, 4, 1)

# Treinar o modelo com os hiperparâmetros ajustados
if st.sidebar.button('Treinar Modelo'):
    rf_model, scaler, mae, mse, r2, y_test, y_pred, feature_importances, feature_names = train_random_forest(n_estimators, max_depth, min_samples_leaf)
    
    # Exibir as métricas
    st.subheader('Métricas de Desempenho do Modelo Treinado')
    st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
    st.write(f"Mean Squared Error (MSE): {mse:.2f}")
    st.write(f"R² Score: {r2:.2f}")
    
    # Exibir previsões e valores reais
    st.subheader('Previsões vs Valores Verdadeiros')
    resultado_df = pd.DataFrame({'Real': y_test, 'Previsto': y_pred})
    st.write(resultado_df.head(20))
    
    # Gráfico de Importância das Features
    st.subheader('Importância dos gases medidos para a predição')
    feature_importance_df = pd.DataFrame({
        'Características': feature_names,
        'Importância': feature_importances
    }).sort_values(by='Importância', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(feature_importance_df['Características'], feature_importance_df['Importância'])
    ax.set_xlabel('Importância')
    ax.set_title('Importância das Características no Random Forest')
    plt.gca().invert_yaxis()  
    st.pyplot(fig)


st.sidebar.header('Entrar Dados para Previsão')

hydrogen = st.sidebar.number_input('Hydrogen', min_value=0.0, value=0.0)
oxygen = st.sidebar.number_input('Oxygen', min_value=0.0, value=0.0)
nitrogen = st.sidebar.number_input('Nitrogen', min_value=0.0, value=0.0)
methane = st.sidebar.number_input('Methane', min_value=0.0, value=0.0)
co = st.sidebar.number_input('CO', min_value=0.0, value=0.0)
co2 = st.sidebar.number_input('CO2', min_value=0.0, value=0.0)
ethylene = st.sidebar.number_input('Ethylene', min_value=0.0, value=0.0)
ethane = st.sidebar.number_input('Ethane', min_value=0.0, value=0.0)
acetylene = st.sidebar.number_input('Acetylene', min_value=0.0, value=0.0)
dbds = st.sidebar.number_input('DBDS', min_value=0.0, value=0.0)
power_factor = st.sidebar.number_input('Power Factor', min_value=0.0, value=0.0)
interfacial_v = st.sidebar.number_input('Interfacial V', min_value=0.0, value=0.0)
dielectric_rigidity = st.sidebar.number_input('Dielectric Rigidity', min_value=0.0, value=0.0)
water_content = st.sidebar.number_input('Water Content', min_value=0.0, value=0.0)


if st.sidebar.button('Fazer Previsão'):
    input_data = np.array([[hydrogen, oxygen, nitrogen, methane, co, co2, ethylene, ethane, acetylene, dbds, power_factor, interfacial_v, dielectric_rigidity, water_content]])
    prediction = load_model_and_predict(input_data)
    st.subheader('Resultado da Previsão')
    st.write(f'Íncide do transformador é: {prediction[0]:.2f}%')
