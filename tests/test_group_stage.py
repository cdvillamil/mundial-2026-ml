import numpy as np

from src.simulation.group_stage import rank_best_thirds, rank_group


def _results(scores):
    """scores: dict (home, away) -> (hg, ag). Devuelve lista de dicts de partido."""
    return [{"home": h, "away": a, "hg": hg, "ag": ag}
            for (h, a), (hg, ag) in scores.items()]


def test_rank_group_by_points():
    # A gana todo, D pierde todo
    teams = ["A", "B", "C", "D"]
    scores = {
        ("A", "B"): (2, 0), ("A", "C"): (2, 0), ("A", "D"): (2, 0),
        ("B", "C"): (1, 0), ("B", "D"): (1, 0), ("C", "D"): (1, 0),
    }
    rng = np.random.default_rng(0)
    order = rank_group(teams, _results(scores), rng)
    assert order[0] == "A"
    assert order[3] == "D"


def test_rank_group_goal_difference_breaks_tie():
    # A y B con mismos puntos; A con mejor diferencia de gol
    teams = ["A", "B", "C", "D"]
    scores = {
        ("A", "B"): (0, 0), ("A", "C"): (5, 0), ("A", "D"): (0, 0),
        ("B", "C"): (1, 0), ("B", "D"): (0, 0), ("C", "D"): (0, 0),
    }
    rng = np.random.default_rng(0)
    order = rank_group(teams, _results(scores), rng)
    assert order.index("A") < order.index("B")  # A arriba por DG


def test_rank_group_head_to_head_breaks_tie():
    # A y B empatan en pts/DG/GF globales; A le gano a B -> A arriba
    teams = ["A", "B", "C", "D"]
    scores = {
        ("A", "B"): (1, 0),   # head to head A>B
        ("A", "C"): (0, 3), ("A", "D"): (3, 0),
        ("B", "C"): (3, 0), ("B", "D"): (0, 3),
        ("C", "D"): (1, 1),
    }
    rng = np.random.default_rng(0)
    order = rank_group(teams, _results(scores), rng)
    assert order.index("A") < order.index("B")


def test_rank_best_thirds_takes_top_n():
    thirds = [
        {"team": "T1", "points": 6, "gd": 3, "gf": 5},
        {"team": "T2", "points": 4, "gd": 1, "gf": 3},
        {"team": "T3", "points": 3, "gd": 0, "gf": 2},
        {"team": "T4", "points": 1, "gd": -2, "gf": 1},
    ]
    rng = np.random.default_rng(0)
    best = rank_best_thirds(thirds, n=2, rng=rng)
    assert best == ["T1", "T2"]
