import pandas as pd
import pytest

from src.features.elo import compute_elo, expected_score, goal_multiplier, k_factor


def test_k_factor():
    assert k_factor("FIFA World Cup") == 60
    assert k_factor("Copa América") == 50
    assert k_factor("UEFA Euro") == 50
    assert k_factor("FIFA World Cup qualification") == 40
    assert k_factor("UEFA Nations League") == 30
    assert k_factor("Friendly") == 20


def test_goal_multiplier():
    assert goal_multiplier(0) == 1.0
    assert goal_multiplier(1) == 1.0
    assert goal_multiplier(2) == 1.5
    assert goal_multiplier(3) == 1.75
    assert goal_multiplier(5) == pytest.approx(2.0)


def test_expected_score_neutral_equal_ratings():
    assert expected_score(1500, 1500, neutral=True) == pytest.approx(0.5)


def test_expected_score_home_advantage():
    assert expected_score(1500, 1500, neutral=False) == pytest.approx(0.640065, abs=1e-5)


def _one_match(neutral, home_score=1, away_score=0, tournament="Friendly"):
    return pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01"]),
        "home_team": ["A"], "away_team": ["B"],
        "home_score": [home_score], "away_score": [away_score],
        "tournament": [tournament], "neutral": [neutral],
    })


def test_compute_elo_hand_calculated_neutral_friendly():
    history, ratings = compute_elo(_one_match(neutral=True))
    # We=0.5, K=20, G=1.0 -> delta = 20*1*(1-0.5) = 10
    assert ratings["A"] == pytest.approx(1510.0)
    assert ratings["B"] == pytest.approx(1490.0)
    assert len(history) == 2  # una fila por equipo


def test_compute_elo_hand_calculated_home_win_by_two():
    _, ratings = compute_elo(_one_match(neutral=False, home_score=2))
    # We=0.640065, K=20, G=1.5 -> delta = 30*(1-0.640065) = 10.798
    assert ratings["A"] == pytest.approx(1510.798, abs=1e-3)
    assert ratings["B"] == pytest.approx(1489.202, abs=1e-3)
