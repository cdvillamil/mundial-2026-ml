import itertools

import pandas as pd

from src.evaluation.historical import extract_groups


def _synthetic_group_stage():
    """2 grupos de 3 equipos (mini) que juegan todos contra todos."""
    rows = []
    g1 = ["A", "B", "C"]
    g2 = ["D", "E", "F"]
    for grp in (g1, g2):
        for h, a in itertools.combinations(grp, 2):
            rows.append({"home_team": h, "away_team": a, "home_goals": 1, "away_goals": 0})
    return pd.DataFrame(rows)


def test_extract_groups_finds_components():
    groups = extract_groups(_synthetic_group_stage(), group_size=3)
    # 2 grupos de 3
    assert len(groups) == 2
    sets = sorted([sorted(g) for g in groups.values()])
    assert sets == [["A", "B", "C"], ["D", "E", "F"]]
