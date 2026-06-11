"""Carga de datos para el dashboard."""
import pandas as pd
import yaml

from src.config import CONFIGS_DIR, PROJECT_ROOT


def load_field() -> dict:
    return yaml.safe_load((CONFIGS_DIR / "groups_2026.yaml").read_text(encoding="utf-8"))


def load_predictions() -> pd.DataFrame:
    path = PROJECT_ROOT / "outputs" / "predictions" / "group_stage_2026_latest.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def build_rate_provider():
    """Entrena el modelo y construye el RateProvider para 2026 (matrices on-demand)."""
    from src.simulation.monte_carlo import _load_field_and_model
    from src.simulation.rates import RateProvider
    field, model, penalty_model = _load_field_and_model()
    rp = RateProvider(model, {t: float(e) for t, e in field["elos"].items()},
                      penalty_model=penalty_model)
    return field, rp


def run_simulation(field, rp, n_sims=50000):
    """Simula con el cuadro oficial si el campo es oficial; si no, el generico."""
    from src.simulation.monte_carlo import (simulate_official_2026,
                                            simulate_tournament)
    if field.get("official"):
        return simulate_official_2026(field["groups"], rp, n_sims=n_sims, seed=42)
    return simulate_tournament(field["groups"], rp, n_sims=n_sims,
                               n_qualify_per_group=2, n_best_thirds=8, seed=42)
