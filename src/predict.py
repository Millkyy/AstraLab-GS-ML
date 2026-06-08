"""Função de predição isolada para uso em scripts ou testes."""
from pathlib import Path
import json
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / 'models' / 'best_model.pkl'
ENCODER_PATH = ROOT / 'models' / 'label_encoder.pkl'
META_PATH = ROOT / 'models' / 'metadata.json'

model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)
metadata = json.loads(META_PATH.read_text(encoding='utf-8'))

def predict_priority(sample: dict) -> dict:
    X = pd.DataFrame([sample])
    y_num = model.predict(X)[0]
    label = encoder.inverse_transform([y_num])[0]
    proba = model.predict_proba(X)[0]
    return {
        'priority_label': label,
        'probabilities': {cls: float(p) for cls, p in zip(encoder.classes_, proba)}
    }

if __name__ == '__main__':
    example = {
        'mission_day': 12,
        'terrain_type': 'gelo_suspeito',
        'latitude': -11.5,
        'longitude': 42.1,
        'distance_to_rover_m': 420,
        'battery_remaining_pct': 82,
        'comm_delay_s': 4.5,
        'radiation_msv_h': 1.1,
        'temperature_c': -75,
        'spectral_ice_index': 0.82,
        'spectral_metal_index': 0.34,
        'image_confidence': 0.91,
        'storage_remaining_kg': 5.4,
        'sample_mass_g': 120,
        'slope_deg': 8.0,
    }
    print(predict_priority(example))
