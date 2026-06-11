import pandas as pd

from src.evaluation.evaluate_predictions import evaluate


def _preds():
    return pd.DataFrame({
        "home": ["A", "C"], "away": ["B", "D"],
        "p_home": [0.7, 0.2], "p_draw": [0.2, 0.3], "p_away": [0.1, 0.5],
        "most_likely": ["2-0", "0-1"],
    })


def test_evaluate_with_results():
    results = pd.DataFrame({
        "home": ["A", "C"], "away": ["B", "D"],
        "home_goals": [2, 1], "away_goals": [0, 1],  # A gana 2-0 (exacto), C-D empate
    })
    rep = evaluate(_preds(), results)
    assert rep["n_evaluated"] == 2
    assert rep["exact_score_acc"] == 0.5   # solo A 2-0 exacto
    assert 0 <= rep["rps"] <= 1
    assert 0 <= rep["accuracy_1x2"] <= 1


def test_evaluate_no_results_yet():
    empty = pd.DataFrame(columns=["home", "away", "home_goals", "away_goals"])
    rep = evaluate(_preds(), empty)
    assert rep["n_evaluated"] == 0
    assert rep["status"] == "sin resultados aun"
