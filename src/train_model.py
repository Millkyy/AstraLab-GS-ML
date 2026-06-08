"""
Treinamento e comparação de modelos para o AstraLab.
Modelos: Regressão Logística, Random Forest, MLP e XGBoost.
Pipeline: pré-processamento, engenharia/seleção de atributos, treino, validação, SHAP e salvamento.
"""
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / 'data' / 'astralab_samples_synthetic.csv'
MODELS_DIR = ROOT / 'models'
REPORTS_DIR = ROOT / 'reports'
FIG_DIR = REPORTS_DIR / 'figures'

TARGET = 'priority_label'
DROP_COLS = ['sample_id', 'scientific_value_score']


def load_data():
    if not DATA_PATH.exists():
        from generate_data import generate_dataset
        df = generate_dataset()
        df.to_csv(DATA_PATH, index=False)
    return pd.read_csv(DATA_PATH)


def make_onehot():
    try:
        return OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown='ignore', sparse=False)


def get_feature_names(preprocessor):
    names = []
    for name, transformer, cols in preprocessor.transformers_:
        if name == 'num':
            names.extend(cols)
        elif name == 'cat':
            names.extend(transformer.get_feature_names_out(cols).tolist())
    return np.array(names)


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    X = df.drop(columns=DROP_COLS + [TARGET])
    y_text = df[TARGET]
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_text)

    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = X.select_dtypes(exclude=['object']).columns.tolist()
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_cols),
        ('cat', make_onehot(), categorical_cols),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    models = {
        'LogisticRegression': LogisticRegression(max_iter=400, class_weight='balanced', random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=90, max_depth=9, random_state=42, class_weight='balanced', n_jobs=-1),
        'MLPClassifier': MLPClassifier(hidden_layer_sizes=(32,), max_iter=120, early_stopping=True, random_state=42),
    }
    if HAS_XGB:
        models['XGBoost'] = XGBClassifier(
            n_estimators=60, max_depth=3, learning_rate=0.08,
            subsample=0.92, colsample_bytree=0.90, objective='multi:softprob',
            eval_metric='mlogloss', random_state=42, n_jobs=1
        )

    rows = []
    fitted = {}
    for name, clf in models.items():
        pipe = Pipeline([
            ('preprocess', preprocessor),
            ('select', SelectKBest(score_func=mutual_info_classif, k=12)),
            ('model', clf),
        ])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        rows.append({
            'model': name,
            'accuracy': accuracy_score(y_test, pred),
            'precision_macro': precision_score(y_test, pred, average='macro'),
            'recall_macro': recall_score(y_test, pred, average='macro'),
            'f1_macro': f1_score(y_test, pred, average='macro'),
        })
        fitted[name] = pipe

    metrics = pd.DataFrame(rows).sort_values('f1_macro', ascending=False)
    metrics.to_csv(REPORTS_DIR / 'metrics.csv', index=False)
    best_name = metrics.iloc[0]['model']
    best_model = fitted[best_name]

    y_pred = best_model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=encoder.classes_)
    (REPORTS_DIR / 'classification_report.txt').write_text(report, encoding='utf-8')

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=encoder.classes_)
    disp.plot(values_format='d')
    plt.title(f'Matriz de confusão - {best_name}')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'confusion_matrix.png', dpi=160)
    plt.close()

    joblib.dump(best_model, MODELS_DIR / 'best_model.pkl')
    joblib.dump(encoder, MODELS_DIR / 'label_encoder.pkl')
    meta = {
        'best_model': best_name,
        'target': TARGET,
        'drop_cols': DROP_COLS,
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols,
        'classes': encoder.classes_.tolist(),
        'dataset_shape': list(df.shape),
    }
    (MODELS_DIR / 'metadata.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')

    # SHAP: usa o melhor modelo já treinado. Para regressão logística, usa LinearExplainer; para árvores, TreeExplainer.
    preprocess = best_model.named_steps['preprocess']
    selector = best_model.named_steps['select']
    final_model = best_model.named_steps['model']
    all_features = get_feature_names(preprocess)
    selected_features = all_features[selector.get_support()]
    X_train_sel = selector.transform(preprocess.transform(X_train))
    X_test_sel = selector.transform(preprocess.transform(X_test))

    if HAS_SHAP:
        try:
            sample = X_test_sel[:80]
            if best_name == 'LogisticRegression':
                explainer = shap.LinearExplainer(final_model, X_train_sel[:200])
                shap_values = explainer.shap_values(sample)
            elif best_name in ['RandomForest', 'XGBoost']:
                explainer = shap.TreeExplainer(final_model)
                shap_values = explainer.shap_values(sample)
            else:
                explainer = shap.KernelExplainer(final_model.predict_proba, X_train_sel[:25])
                shap_values = explainer.shap_values(sample[:30])
            if isinstance(shap_values, list):
                values = np.mean([np.abs(v).mean(axis=0) for v in shap_values], axis=0)
            else:
                arr = np.asarray(shap_values)
                if arr.ndim == 3:
                    values = np.abs(arr).mean(axis=(0, 2))
                else:
                    values = np.abs(arr).mean(axis=0)
            values = np.asarray(values).reshape(-1)[:len(selected_features)]
            shap_df = pd.DataFrame({'feature': selected_features, 'mean_abs_shap': values}).sort_values('mean_abs_shap', ascending=False)
        except Exception as e:
            (REPORTS_DIR / 'shap_error.txt').write_text(str(e), encoding='utf-8')
            shap_df = pd.DataFrame({'feature': selected_features, 'mean_abs_shap': getattr(final_model, 'feature_importances_', np.zeros(len(selected_features)))[:len(selected_features)]})
    else:
        # fallback para permitir visualização caso SHAP não esteja instalado
        shap_df = pd.DataFrame({'feature': selected_features, 'mean_abs_shap': getattr(final_model, 'feature_importances_', np.zeros(len(selected_features)))[:len(selected_features)]})

    shap_df = shap_df.sort_values('mean_abs_shap', ascending=False)
    shap_df.to_csv(REPORTS_DIR / 'shap_feature_importance.csv', index=False)
    top = shap_df.head(10).sort_values('mean_abs_shap')
    plt.figure(figsize=(8, 5))
    plt.barh(top['feature'], top['mean_abs_shap'])
    plt.title('Top 10 variáveis por SHAP')
    plt.xlabel('Média |SHAP|')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'shap_top_features.png', dpi=160)
    plt.close()

    print('Métricas:')
    print(metrics)
    print('\nMelhor modelo:', best_name)
    print('\nTop SHAP:')
    print(shap_df.head(10))


if __name__ == '__main__':
    main()
