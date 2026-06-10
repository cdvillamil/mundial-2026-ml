import pandas as pd

from src.features.form import compute_form


def _matches():
    """10 partidos de A (1872-1882) y 0 de B."""
    dates = pd.date_range("1872-01-01", periods=10, freq="30D")
    return pd.DataFrame({
        "date": dates,
        "home_team": ["A"] * 10,
        "away_team": ["B"] * 10,
        "home_goals": [1, 2, 1, 1, 0, 1, 2, 1, 0, 1],
        "away_goals": [0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
    })


def test_form_features_are_empty_before_window():
    """Las primeras 5 filas no tienen forma (ventana vacía)."""
    df = _matches()
    form = compute_form(df, "home_team", windows=[5, 10])
    assert form.iloc[0]["form_5_wins"] is pd.NA or pd.isna(form.iloc[0]["form_5_wins"])
    assert form.iloc[4]["form_5_wins"] is pd.NA or pd.isna(form.iloc[4]["form_5_wins"])
    assert not (form.iloc[5]["form_5_wins"] is pd.NA or pd.isna(form.iloc[5]["form_5_wins"]))


def test_form_5_window_calculates_correctly():
    df = _matches()
    form = compute_form(df, "home_team", windows=[5])
    # Primeros 5 partidos de A (índices 0-4): W W W D D (3 victorias, 2 empates)
    assert form.iloc[5]["form_5_wins"] == 3
    assert form.iloc[5]["form_5_draws"] == 2
    assert form.iloc[5]["form_5_losses"] == 0
    assert form.iloc[5]["form_5_gf"] == 5  # 1+2+1+1+0
    assert form.iloc[5]["form_5_ga"] == 1  # 0+0+0+1+0
