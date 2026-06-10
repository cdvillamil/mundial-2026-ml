import pandas as pd

from src.features.context import compute_context_features


def _matches():
    return pd.DataFrame({
        "date": pd.to_datetime(["2022-12-09", "2022-12-18"]),
        "home_team": ["A", "A"],
        "away_team": ["B", "C"],
        "neutral": [False, True],
    })


def test_context_features_home_advantage():
    df = _matches()
    ctx = compute_context_features(df, "home_team")
    # Primer partido: no neutral → is_neutral=0
    assert ctx.iloc[0]["is_neutral"] == 0
    assert ctx.iloc[1]["is_neutral"] == 1


def test_days_rest():
    df = _matches()
    ctx = compute_context_features(df, "home_team")
    # Segundo partido de A es 9 días después del primero
    assert ctx.iloc[1]["days_since_last"] == 9
    assert pd.isna(ctx.iloc[0]["days_since_last"])
