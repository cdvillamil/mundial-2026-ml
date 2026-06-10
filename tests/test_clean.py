import pandas as pd

from src.etl.clean import build_alias_map, clean_results


def _former_names():
    return pd.DataFrame({
        "current": ["DR Congo", "Indonesia"],
        "former": ["Zaïre", "Dutch East Indies"],
    })


def _raw_results():
    return pd.DataFrame({
        "date": ["2022-12-18", "2022-12-18", "1974-09-22", "bad-date"],
        "home_team": ["Argentina", "Argentina", "Zaïre", "Spain"],
        "away_team": ["France", "France", "Ghana", "France"],
        "home_score": [3, 3, 2, 1],
        "away_score": [3, 3, 1, 0],
        "tournament": ["FIFA World Cup"] * 3 + ["Friendly"],
        "city": ["Lusail", "Lusail", "Kinshasa", "Madrid"],
        "country": ["Qatar", "Qatar", "Zaire", "Spain"],
        "neutral": [True, True, False, False],
    })


def test_alias_map_maps_former_to_current():
    assert build_alias_map(_former_names())["Zaïre"] == "DR Congo"


def test_clean_results_dedups_normalizes_and_drops_bad_rows():
    out = clean_results(_raw_results(), build_alias_map(_former_names()))
    assert len(out) == 2                      # dedup + fila con fecha invalida fuera
    assert "Zaïre" not in set(out["home_team"])
    assert "DR Congo" in set(out["home_team"])
    assert out["date"].is_monotonic_increasing
    assert out["home_score"].dtype.kind == "i"
    assert out["neutral"].dtype == bool
