# AstraLab ML — Priorização Inteligente de Amostras Espaciais

## 1. Contexto do problema

O **AstraLab** é uma plataforma educacional para simular uma missão espacial de coleta, triagem e análise de amostras em ambientes extremos, como Lua, Marte, asteroides ou estações remotas de pesquisa. O problema tratado neste projeto é:

> Como prever a prioridade de coleta de uma amostra espacial considerando valor científico, risco operacional e recursos limitados da missão?

Em uma missão espacial simulada, coletar uma amostra pouco relevante pode desperdiçar bateria, tempo, comunicação e armazenamento. Por isso, este projeto cria um pipeline de Machine Learning para classificar amostras em **Baixa**, **Media** ou **Alta** prioridade de coleta.

## 2. Fonte dos dados

Foi criado um **dataset sintético** usando IA/geração programática com regras coerentes com o domínio do AstraLab. O arquivo final está em:

```text
data/astralab_samples_synthetic.csv
```

O dataset possui **1.200 linhas** e **18 colunas**.

Principais variáveis:

- `terrain_type`: tipo de terreno/amostra identificado.
- `distance_to_rover_m`: distância até o robô.
- `battery_remaining_pct`: bateria disponível.
- `comm_delay_s`: atraso de comunicação.
- `radiation_msv_h`: nível de radiação.
- `temperature_c`: temperatura do ambiente.
- `spectral_ice_index`: indício espectral de gelo.
- `spectral_metal_index`: indício espectral metálico.
- `image_confidence`: confiança da visão computacional.
- `storage_remaining_kg`: armazenamento disponível.
- `sample_mass_g`: massa estimada da amostra.
- `slope_deg`: inclinação do terreno.
- `priority_label`: alvo de classificação.

A coluna `scientific_value_score` foi usada apenas para gerar o rótulo sintético e foi removida do treinamento para evitar vazamento de dados.

## 3. Metodologia utilizada

O pipeline contempla:

1. Geração do dataset sintético.
2. Separação treino/teste com `train_test_split` e estratificação.
3. Pré-processamento:
   - `StandardScaler` para variáveis numéricas.
   - `OneHotEncoder` para variáveis categóricas.
4. Engenharia e seleção de atributos:
   - uso de variáveis operacionais e científicas;
   - `SelectKBest` com `mutual_info_classif`.
5. Treinamento e comparação de modelos.
6. Escolha do melhor modelo com base em `f1_macro`.
7. Interpretabilidade com SHAP.
8. Deploy local via Streamlit.

## 4. Modelos testados

Foram testadas quatro técnicas:

- Regressão Logística;
- Random Forest;
- MLP Classifier;
- XGBoost Classifier.


## 5. Resultados obtidos

As métricas são salvas em:

```text
reports/metrics.csv
reports/classification_report.txt
reports/figures/confusion_matrix.png
```

Métricas utilizadas:

- Accuracy;
- Precision macro;
- Recall macro;
- F1 macro.

O melhor modelo é salvo em:

```text
models/best_model.pkl
```

## 6. Interpretação com SHAP

A interpretabilidade foi feita com SHAP e os resultados são salvos em:

```text
reports/shap_feature_importance.csv
reports/figures/shap_top_features.png
```

O SHAP permite explicar quais variáveis mais influenciaram a decisão do modelo. No contexto do AstraLab, espera-se que variáveis como índice espectral de gelo, índice metálico, bateria, distância até o rover, radiação, atraso de comunicação e massa da amostra estejam entre as mais relevantes.

## 7. Como executar o projeto

### 7.1 Criar ambiente e instalar dependências

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### 7.2 Gerar os dados

```bash
python src/generate_data.py
```

### 7.3 Treinar e avaliar os modelos

```bash
python src/train_model.py
```

### 7.4 Rodar a aplicação Streamlit

```bash
streamlit run app.py
```

## 8. Deploy

A aplicação foi publicada no **Streamlit Community Cloud**.

Link da aplicação em funcionamento:

```text
https://astralab-gs-ml-app.streamlit.app/
```

Link do repositório GitHub:

```text
https://github.com/Millkyy/AstraLab-GS-ML
```

## 8. Estrutura do repositório

```text
astralab_ml_project/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── astralab_samples_synthetic.csv
├── models/
│   ├── best_model.pkl
│   ├── label_encoder.pkl
│   └── metadata.json
├── reports/
│   ├── metrics.csv
│   ├── classification_report.txt
│   ├── shap_feature_importance.csv
│   ├── checklist_avaliacao.md
│   └── figures/
│       ├── confusion_matrix.png
│       └── shap_top_features.png
└── src/
    ├── generate_data.py
    ├── train_model.py
    └── predict.py
```

## 9. Integrantes

- Aline Fernandes Zeppelini - RM 97966 
- Camilly Breitbach Ishida - RM 551474 
- Julia Leite Galvão - RM 550201 
