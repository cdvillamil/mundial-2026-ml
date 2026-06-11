import pandas as pd

from src.features.elo_features import attach_pre_match_elo, tournament_importance


def test_tournament_importance_levels():
    assert tournament_importance("Friendly") == 0
    assert tournament_importance("FIFA World Cup qualification") == 1
    assert tournament_importance("Copa América") == 2
    assert tournament_importance("FIFA World Cup") == 3


def test_attach_pre_match_elo_is_point_in_time():
    # elo_history: A sube de 1500 a 1600 tras un partido el 2020-01-10
    elo_history = pd.DataFrame({
        "team": ["A", "B", "A"],
        "date": pd.to_datetime(["2020-01-10", "2020-01-10", "2020-06-01"]),
        "elo": [1600.0, 1400.0, 1650.0],
    })
    matches = pd.DataFrame({
        "date": pd.to_datetime(["2020-03-01"]),
        "home_team": ["A"], "away_team": ["B"],
        "tournament": ["Friendly"],
    })
    out = attach_pre_match_elo(matches, elo_history)
    # En 2020-03-01, el ultimo Elo de A previo es 1600 (no 1650, que es de junio)
    assert out.iloc[0]["home_elo"] == 1600.0
    assert out.iloc[0]["away_elo"] == 1400.0
    assert out.iloc[0]["elo_diff"] == 200.0


def test_attach_handles_missing_elo_with_default():
    elo_history = pd.DataFrame({"team": ["A"], "date": pd.to_datetime(["2020-01-10"]), "elo": [1600.0]})
    matches = pd.DataFrame({
        "date": pd.to_datetime(["2020-03-01"]),
        "home_team": ["A"], "away_team": ["NEW"],
        "tournament": ["Friendly"],
    })
    out = attach_pre_match_elo(matches, elo_history)
    assert out.iloc[0]["away_elo"] == 1500.0  # default para equipo sin historia
