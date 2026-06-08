"""
Geração de dataset sintético para o AstraLab.
Tema: priorização inteligente de coleta de amostras espaciais.
Saída: data/astralab_samples_synthetic.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd

RANDOM_STATE = 42
N_ROWS = 1200

def generate_dataset(n_rows: int = N_ROWS, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    terrain_types = np.array(['regolitho', 'cratera', 'basalto', 'gelo_suspeito', 'metalico', 'poeira_fina'])
    terrain = rng.choice(terrain_types, size=n_rows, p=[0.27,0.18,0.18,0.12,0.10,0.15])

    mission_day = rng.integers(1, 31, size=n_rows)
    latitude = rng.uniform(-35, 35, size=n_rows).round(5)
    longitude = rng.uniform(-180, 180, size=n_rows).round(5)
    distance_to_rover_m = np.clip(rng.gamma(shape=2.2, scale=380, size=n_rows), 20, 2800).round(2)
    battery_remaining_pct = np.clip(rng.normal(62, 18, size=n_rows), 12, 100).round(2)
    comm_delay_s = np.clip(rng.normal(6.5, 2.2, size=n_rows) + distance_to_rover_m/1400, 1.0, 16.0).round(2)
    radiation_msv_h = np.clip(rng.lognormal(mean=0.45, sigma=0.45, size=n_rows), 0.2, 8.5).round(3)
    temperature_c = np.clip(rng.normal(-58, 28, size=n_rows), -145, 35).round(2)
    spectral_ice_index = np.clip(rng.beta(2.0, 4.0, size=n_rows), 0, 1)
    spectral_metal_index = np.clip(rng.beta(2.3, 3.8, size=n_rows), 0, 1)
    image_confidence = np.clip(rng.normal(0.79, 0.11, size=n_rows), 0.42, 0.99)
    storage_remaining_kg = np.clip(rng.normal(4.3, 1.4, size=n_rows), 0.4, 8.0).round(2)
    sample_mass_g = np.clip(rng.normal(145, 55, size=n_rows), 25, 330).round(1)
    slope_deg = np.clip(rng.normal(11, 7, size=n_rows), 0, 38).round(2)

    # Ajustes coerentes por terreno
    spectral_ice_index += np.where(terrain == 'gelo_suspeito', rng.normal(0.34, 0.08, n_rows), 0)
    spectral_metal_index += np.where(terrain == 'metalico', rng.normal(0.30, 0.10, n_rows), 0)
    radiation_msv_h += np.where(terrain == 'cratera', rng.normal(0.75, 0.25, n_rows), 0)
    slope_deg += np.where(terrain == 'cratera', rng.normal(4.0, 2.0, n_rows), 0)
    spectral_ice_index = np.clip(spectral_ice_index, 0, 1).round(3)
    spectral_metal_index = np.clip(spectral_metal_index, 0, 1).round(3)
    radiation_msv_h = np.clip(radiation_msv_h, 0.2, 9.5).round(3)
    slope_deg = np.clip(slope_deg, 0, 42).round(2)

    terrain_bonus = {
        'gelo_suspeito': 18,
        'metalico': 15,
        'cratera': 8,
        'basalto': 6,
        'regolitho': 2,
        'poeira_fina': 0,
    }
    bonus = np.array([terrain_bonus[t] for t in terrain])

    # Score operacional + científico. Regras simulariam o julgamento da equipe de missão.
    score = (
        42 * spectral_ice_index +
        32 * spectral_metal_index +
        0.23 * battery_remaining_pct +
        16 * image_confidence +
        bonus -
        0.010 * distance_to_rover_m -
        1.75 * comm_delay_s -
        2.35 * radiation_msv_h -
        0.40 * slope_deg -
        0.020 * sample_mass_g +
        1.7 * storage_remaining_kg +
        rng.normal(0, 4.2, size=n_rows)
    )
    scientific_value_score = np.clip(score, 0, 100).round(2)

    # qcut gera 3 classes balanceadas, inspirado na aula de classificação que dividia dados em Low/Medium/High.
    priority_label = pd.qcut(scientific_value_score, q=3, labels=['Baixa', 'Media', 'Alta']).astype(str)

    df = pd.DataFrame({
        'sample_id': [f'AST-{i:04d}' for i in range(1, n_rows + 1)],
        'mission_day': mission_day,
        'terrain_type': terrain,
        'latitude': latitude,
        'longitude': longitude,
        'distance_to_rover_m': distance_to_rover_m,
        'battery_remaining_pct': battery_remaining_pct,
        'comm_delay_s': comm_delay_s,
        'radiation_msv_h': radiation_msv_h,
        'temperature_c': temperature_c,
        'spectral_ice_index': spectral_ice_index,
        'spectral_metal_index': spectral_metal_index,
        'image_confidence': image_confidence.round(3),
        'storage_remaining_kg': storage_remaining_kg,
        'sample_mass_g': sample_mass_g,
        'slope_deg': slope_deg,
        'scientific_value_score': scientific_value_score,
        'priority_label': priority_label,
    })
    return df

if __name__ == '__main__':
    out = Path(__file__).resolve().parents[1] / 'data' / 'astralab_samples_synthetic.csv'
    df = generate_dataset()
    df.to_csv(out, index=False)
    print(f'Dataset salvo em: {out}')
    print(df.shape)
    print(df.head())
