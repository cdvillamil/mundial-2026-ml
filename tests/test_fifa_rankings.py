import pandas as pd

from src.etl.fifa_rankings import tidy_rankings


def test_tidy_rankings_normalizes_columns_and_names():
    raw = pd.DataFrame({
        "rank": [1, 2],
        "country_full": ["Argentina", "IR Iran"],
        "total_points": [1860.1, 1600.0],
        "rank_date": ["2024-04-04", "2024-04-04"],
    })
    out = tidy_rankings(raw, alias_map={"IR Iran": "Iran"})
    assert list(out.columns) == ["team_name", "ranking_date", "rank", "points"]
    assert set(out["team_name"]) == {"Argentina", "Iran"}
