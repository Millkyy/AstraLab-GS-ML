# Checklist de avaliação — AstraLab ML

| Critério de avaliação | Peso | Evidência no projeto | Status |
|---|---:|---|---|
| Definição do problema e qualidade dos dados | 15 | Problema descrito no README; dataset sintético com 1.200 linhas e 18 colunas em `data/astralab_samples_synthetic.csv`. | Cumprido |
| Pré-processamento e engenharia de atributos | 20 | `ColumnTransformer`, `StandardScaler`, `OneHotEncoder`, remoção de vazamento de dados e `SelectKBest` em `src/train_model.py`. | Cumprido |
| Aplicação e comparação de modelos | 20 | Regressão Logística, Random Forest, MLP e XGBoost em `src/train_model.py`. | Cumprido |
| Validação e análise de métricas | 15 | `accuracy`, `precision_macro`, `recall_macro`, `f1_macro`, matriz de confusão e relatório de classificação em `reports/`. | Cumprido |
| Interpretabilidade com SHAP | 10 | `reports/shap_feature_importance.csv` e `reports/figures/shap_top_features.png`. | Cumprido |
| Deploy da aplicação | 10 | `app.py` em Streamlit pronto para execução e publicação. | Parcial: falta publicar online |
| Organização do código e README no GitHub | 10 | Estrutura pronta com `README.md`, `requirements.txt`, `src/`, `data/`, `models/`, `reports/`. | Parcial: falta subir no GitHub |

## Observação final

Tudo que depende de código, dataset, pipeline, modelos, métricas e aplicação está pronto. Para a entrega final da disciplina, ainda é necessário publicar o repositório no GitHub e fazer o deploy online da aplicação para obter os dois links exigidos.
