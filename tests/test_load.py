import sqlite3

import pandas as pd

from src.etl.load import load_matches


def _matches():
    return pd.DataFrame({
        "date": pd.to_datetime(["2022-12-18", "2022-12-09"]),
        "home_team": ["Argentina", "Netherlands"],
        "away_team": ["France", "Argentina"],
        "home_score": [3, 2],
        "away_score": [3, 2],
        "tournament": ["FIFA World Cup", "FIFA World Cup"],
        "city": ["Lusail", "Lusail"],
        "country": ["Qatar", "Qatar"],
        "neutral": [True, True],
    })


def _shootouts():
    return pd.DataFrame({
        "date": pd.to_datetime(["2022-12-18", "2022-12-09"]),
        "home_team": ["Argentina", "Netherlands"],
        "away_team": ["France", "Argentina"],
        "winner": ["Argentina", "Argentina"],
    })


def test_load_matches_builds_teams_and_links_shootouts(tmp_path):
    db = tmp_path / "test.sqlite"
    n = load_matches(_matches(), _shootouts(), db_path=db, processed_dir=tmp_path)
    assert n == 2
    con = sqlite3.connect(db)
    teams = pd.read_sql("SELECT * FROM teams", con)
    assert set(teams["name"]) == {"Argentina", "France", "Netherlands"}
    m = pd.read_sql("""
        SELECT m.*, t.name AS winner_name FROM matches m
        LEFT JOIN teams t ON t.team_id = m.shootout_winner_id
    """, con)
    con.close()
    assert set(m["winner_name"]) == {"Argentina"}
    assert (tmp_path / "matches.parquet").exists()
