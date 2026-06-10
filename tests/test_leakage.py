"""Verifica que ninguna feature "conoce el futuro" de un partido."""
import pandas as pd

from src.features.context import compute_context_features
from src.features.form import compute_form


def _matches_with_future():
    """3 partidos; testeamos que features en t no cambien si removemos t+1."""
    dates = pd.to_datetime(["2022-01-01", "2022-01-15", "2022-02-01"])
    return pd.DataFrame({
        "date": dates,
        "home_team": ["A", "A", "B"],
        "away_team": ["B", "C", "A"],
        "home_goals": [1, 2, 1],
        "away_goals": [0, 0, 0],
        "neutral": [False, False, False],
    })


def test_form_features_frozen_at_match_date():
    full = _matches_with_future()
    form_full = compute_form(full, "home_team", windows=[5])

    # Remover el tercer partido
    truncated = full.iloc[:2]
    form_trunc = compute_form(truncated, "home_team", windows=[5])

    # La forma de A en la fecha del 2o partido no debe cambiar
    assert form_full.iloc[1]["form_5_wins"] == form_trunc.iloc[1]["form_5_wins"]
    assert form_full.iloc[1]["form_5_gf"] == form_trunc.iloc[1]["form_5_gf"]


def test_context_features_frozen_at_match_date():
    full = _matches_with_future()
    ctx_full = compute_context_features(full, "home_team")

    truncated = full.iloc[:2]
    ctx_trunc = compute_context_features(truncated, "home_team")

    # El contexto del 2o partido es igual con o sin el 3o
    assert ctx_full.iloc[1]["is_neutral"] == ctx_trunc.iloc[1]["is_neutral"]
    assert ctx_full.iloc[1]["days_since_last"] == ctx_trunc.iloc[1]["days_since_last"]
