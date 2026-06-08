import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / 'models' / 'best_model.pkl'
ENCODER_PATH = ROOT / 'models' / 'label_encoder.pkl'
META_PATH = ROOT / 'models' / 'metadata.json'
METRICS_PATH = ROOT / 'reports' / 'metrics.csv'
SHAP_PATH = ROOT / 'reports' / 'shap_feature_importance.csv'

st.set_page_config(
    page_title='AstraLab ML',
    page_icon='🛰️',
    layout='wide'
)

# ---------- CSS ----------
st.markdown("""
<style>
.main-title {
    font-size: 3rem;
    font-weight: 800;
    color: #1f2937;
}
.two-title {
    font-size: 2rem;
    font-weight: 500;
    color: #1f2937;
    margin-bottom: 0.2rem;
}
.subtitle {
    font-size: 1.1rem;
    color: #4b5563;
    margin-bottom: 1.5rem;
}
.card {
    border-radius: 18px;
    padding: 24px;
    color: white;
    box-shadow: 0 6px 18px rgba(0,0,0,0.12);
    margin-bottom: 1rem;
}
.card h3 {
    margin: 0;
    font-size: 1rem;
    opacity: 0.95;
}
.card h1 {
    margin: 0.4rem 0 0 0;
    font-size: 2.8rem;
    font-weight: 800;
}
.high {
    background: linear-gradient(135deg, #16a34a, #22c55e);
}
.medium {
    background: linear-gradient(135deg, #ea580c, #f59e0b);
}
.low {
    background: linear-gradient(135deg, #dc2626, #ef4444);
}
.info-box {
    background: #f8fafc;
    border: 1px solid #e5e7eb;
    padding: 16px;
    border-radius: 14px;
}
.small-muted {
    color: #6b7280;
    font-size: 0.95rem;
}
</style>
""", unsafe_allow_html=True)

# ---------- carregamento ----------
@st.cache_resource
def load_assets():
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    metadata = json.loads(META_PATH.read_text(encoding='utf-8'))
    return model, encoder, metadata

model, encoder, metadata = load_assets()

# ---------- funções auxiliares ----------
def priority_class(priority):
    p = priority.lower()
    if p == "alta":
        return "high"
    elif p == "media" or p == "média":
        return "medium"
    return "low"

def priority_explanation(priority):
    p = priority.lower()
    if p == "alta":
        return "A amostra apresenta forte potencial científico e/ou boa viabilidade operacional para coleta."
    elif p == "media" or p == "média":
        return "A amostra tem relevância moderada. A coleta pode ser interessante dependendo das condições da missão."
    return "A amostra possui baixa prioridade no momento, seja por menor valor científico ou maior custo/risco operacional."

# ---------- título ----------
st.markdown('<div class="main-title">🛰️ AstraLab</div>', unsafe_allow_html=True)
st.markdown('<div class="two-title">Priorização Inteligente de Amostras Espaciais</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Aplicação Streamlit para prever a prioridade de coleta de uma amostra espacial simulada.</div>', unsafe_allow_html=True)

# ---------- sidebar ----------
st.sidebar.title("🔧 Configuração da amostra")

with st.sidebar.expander("🌍 Localização e missão", expanded=True):
    terrain_type = st.selectbox(
        'Tipo de terreno',
        ['regolitho', 'cratera', 'basalto', 'gelo_suspeito', 'metalico', 'poeira_fina'],
        index=3
    )
    mission_day = st.slider('Dia da missão', 1, 30, 12)
    latitude = st.number_input('Latitude', value=-11.5)
    longitude = st.number_input('Longitude', value=42.1)

with st.sidebar.expander("🤖 Condições operacionais", expanded=True):
    distance_to_rover_m = st.slider('Distância até o rover (m)', 20.0, 2800.0, 420.0)
    battery_remaining_pct = st.slider('Bateria restante (%)', 12.0, 100.0, 82.0)
    comm_delay_s = st.slider('Atraso de comunicação (s)', 1.0, 16.0, 4.5)
    radiation_msv_h = st.slider('Radiação (mSv/h)', 0.2, 9.5, 1.1)
    temperature_c = st.slider('Temperatura (°C)', -145.0, 35.0, -75.0)
    storage_remaining_kg = st.slider('Armazenamento restante (kg)', 0.4, 8.0, 5.4)
    slope_deg = st.slider('Inclinação do terreno (graus)', 0.0, 42.0, 8.0)

with st.sidebar.expander("🧪 Indicadores científicos", expanded=True):
    spectral_ice_index = st.slider('Índice espectral de gelo', 0.0, 1.0, 0.82)
    spectral_metal_index = st.slider('Índice espectral metálico', 0.0, 1.0, 0.34)
    image_confidence = st.slider('Confiança da visão computacional', 0.42, 0.99, 0.91)
    sample_mass_g = st.slider('Massa da amostra (g)', 25.0, 330.0, 120.0)

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

prob_df = pd.DataFrame({
    'Classe': encoder.classes_,
    'Probabilidade': proba
}).sort_values('Probabilidade', ascending=False).reset_index(drop=True)

# ---------- abas ----------
tab1, tab2, tab3 = st.tabs(["🔮 Previsão", "📊 Validação dos modelos", "🧠 SHAP"])

# ---------- aba 1 ----------
with tab1:
    col1, col2 = st.columns([1.1, 0.9])

    with col1:
        st.subheader("Resultado principal")

        css_class = priority_class(priority)
        st.markdown(
            f"""
            <div class="card {css_class}">
                <h3>Prioridade prevista</h3>
                <h1>{priority}</h1>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="info-box">
                <b>Interpretação:</b><br>
                {priority_explanation(priority)}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("#### Probabilidade por classe")
        for _, row in prob_df.iterrows():
            classe = row["Classe"]
            prob = float(row["Probabilidade"])
            st.write(f"**{classe}** — {prob:.2%}")
            st.progress(int(prob * 100))

        st.caption("Classes possíveis: Baixa, Media e Alta prioridade de coleta.")

    with col2:
        st.subheader("Resumo da amostra")
        resumo_df = pd.DataFrame({
            "Variável": [
                "Tipo de terreno",
                "Dia da missão",
                "Latitude",
                "Longitude",
                "Distância até o rover (m)",
                "Bateria restante (%)",
                "Atraso de comunicação (s)",
                "Radiação (mSv/h)",
                "Temperatura (°C)",
                "Índice espectral de gelo",
                "Índice espectral metálico",
                "Confiança da visão computacional",
                "Armazenamento restante (kg)",
                "Massa da amostra (g)",
                "Inclinação do terreno (graus)"
            ],
            "Valor": [
                terrain_type,
                mission_day,
                latitude,
                longitude,
                distance_to_rover_m,
                battery_remaining_pct,
                comm_delay_s,
                radiation_msv_h,
                temperature_c,
                spectral_ice_index,
                spectral_metal_index,
                image_confidence,
                storage_remaining_kg,
                sample_mass_g,
                slope_deg
            ]
        })
        st.dataframe(resumo_df, use_container_width=True, hide_index=True)

# ---------- aba 2 ----------
with tab2:
    st.subheader("Comparação de desempenho dos modelos")
    st.write("Esta seção mostra o desempenho dos modelos testados durante o treinamento.")

    if METRICS_PATH.exists():
        metrics_df = pd.read_csv(METRICS_PATH)
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

        if not metrics_df.empty:
            best_row = metrics_df.sort_values("f1_macro", ascending=False).iloc[0]
            st.success(
                f"Melhor modelo: **{best_row['model']}** | "
                f"Accuracy: **{best_row['accuracy']:.4f}** | "
                f"F1 Macro: **{best_row['f1_macro']:.4f}**"
            )

    fig_cm = ROOT / 'reports' / 'figures' / 'confusion_matrix.png'
    if fig_cm.exists():
        st.image(str(fig_cm), caption='Matriz de confusão do melhor modelo', use_container_width=True)

# ---------- aba 3 ----------
with tab3:
    st.subheader("Interpretabilidade com SHAP")
    st.write("O SHAP mostra quais variáveis mais influenciaram as decisões do modelo.")

    if SHAP_PATH.exists():
        shap_df = pd.read_csv(SHAP_PATH)
        st.dataframe(shap_df.head(12), use_container_width=True, hide_index=True)

        fig_path = ROOT / 'reports' / 'figures' / 'shap_top_features.png'
        if fig_path.exists():
            st.image(str(fig_path), caption='Top variáveis por SHAP', use_container_width=True)
    else:
        st.warning('Arquivo de SHAP não encontrado. Rode: python src/train_model.py')