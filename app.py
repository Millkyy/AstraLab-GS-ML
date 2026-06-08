import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / 'models' / 'best_model.pkl'
ENCODER_PATH = ROOT / 'models' / 'label_encoder.pkl'
META_PATH = ROOT / 'models' / 'metadata.json'
DATA_PATH = ROOT / 'data' / 'astralab_samples_synthetic.csv'
METRICS_PATH = ROOT / 'reports' / 'metrics.csv'
SHAP_PATH = ROOT / 'reports' / 'shap_feature_importance.csv'

st.set_page_config(page_title='AstraLab ML', page_icon='🛰️', layout='wide')
st.title('🛰️ AstraLab — Priorização Inteligente de Amostras Espaciais')
st.write('Aplicação Streamlit para prever a prioridade de coleta de uma amostra espacial simulada.')

@st.cache_resource
def load_assets():
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    metadata = json.loads(META_PATH.read_text(encoding='utf-8'))
    return model, encoder, metadata

model, encoder, metadata = load_assets()

st.sidebar.header('Dados da amostra')
terrain_type = st.sidebar.selectbox('Tipo de terreno', ['regolitho', 'cratera', 'basalto', 'gelo_suspeito', 'metalico', 'poeira_fina'], index=3)
mission_day = st.sidebar.slider('Dia da missão', 1, 30, 12)
latitude = st.sidebar.number_input('Latitude', value=-11.5)
longitude = st.sidebar.number_input('Longitude', value=42.1)
distance_to_rover_m = st.sidebar.slider('Distância até o rover (m)', 20.0, 2800.0, 420.0)
battery_remaining_pct = st.sidebar.slider('Bateria restante (%)', 12.0, 100.0, 82.0)
comm_delay_s = st.sidebar.slider('Atraso de comunicação (s)', 1.0, 16.0, 4.5)
radiation_msv_h = st.sidebar.slider('Radiação (mSv/h)', 0.2, 9.5, 1.1)
temperature_c = st.sidebar.slider('Temperatura (°C)', -145.0, 35.0, -75.0)
spectral_ice_index = st.sidebar.slider('Índice espectral de gelo', 0.0, 1.0, 0.82)
spectral_metal_index = st.sidebar.slider('Índice espectral metálico', 0.0, 1.0, 0.34)
image_confidence = st.sidebar.slider('Confiança da visão computacional', 0.42, 0.99, 0.91)
storage_remaining_kg = st.sidebar.slider('Armazenamento restante (kg)', 0.4, 8.0, 5.4)
sample_mass_g = st.sidebar.slider('Massa da amostra (g)', 25.0, 330.0, 120.0)
slope_deg = st.sidebar.slider('Inclinação do terreno (graus)', 0.0, 42.0, 8.0)

sample = pd.DataFrame([{
    'mission_day': mission_day,
    'terrain_type': terrain_type,
    'latitude': latitude,
    'longitude': longitude,
    'distance_to_rover_m': distance_to_rover_m,
    'battery_remaining_pct': battery_remaining_pct,
    'comm_delay_s': comm_delay_s,
    'radiation_msv_h': radiation_msv_h,
    'temperature_c': temperature_c,
    'spectral_ice_index': spectral_ice_index,
    'spectral_metal_index': spectral_metal_index,
    'image_confidence': image_confidence,
    'storage_remaining_kg': storage_remaining_kg,
    'sample_mass_g': sample_mass_g,
    'slope_deg': slope_deg,
}])

pred_num = model.predict(sample)[0]
priority = encoder.inverse_transform([pred_num])[0]
proba = model.predict_proba(sample)[0]

col1, col2 = st.columns([1, 1])
with col1:
    st.subheader('Resultado')
    st.metric('Prioridade prevista', priority)
    st.dataframe(pd.DataFrame({'Classe': encoder.classes_, 'Probabilidade': proba}).sort_values('Probabilidade', ascending=False), use_container_width=True)
    st.caption('Classes: Baixa, Media e Alta prioridade de coleta.')
with col2:
    st.subheader('Entrada enviada ao modelo')
    st.dataframe(sample, use_container_width=True)

st.divider()
st.subheader('Validação dos modelos')
if METRICS_PATH.exists():
    st.dataframe(pd.read_csv(METRICS_PATH), use_container_width=True)
if (ROOT / 'reports' / 'figures' / 'confusion_matrix.png').exists():
    st.image(str(ROOT / 'reports' / 'figures' / 'confusion_matrix.png'), caption='Matriz de confusão do melhor modelo')

st.subheader('Interpretabilidade com SHAP')
if SHAP_PATH.exists():
    shap_df = pd.read_csv(SHAP_PATH)
    st.write('Variáveis com maior influência média absoluta nas previsões:')
    st.dataframe(shap_df.head(12), use_container_width=True)
    fig_path = ROOT / 'reports' / 'figures' / 'shap_top_features.png'
    if fig_path.exists():
        st.image(str(fig_path), caption='Top variáveis por SHAP')
else:
    st.warning('Arquivo de SHAP não encontrado. Rode: python src/train_model.py')
