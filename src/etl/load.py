"""Carga de tablas limpias a SQLite y Parquet."""
import sqlite3
from pathlib import Path

import pandas as pd

from src.config import DB_PATH, PROCESSED_DIR

SCHEMA = """
CREATE TABLE IF NOT EXISTS teams (
    team_id INTEGER PRIMARY KEY,
    name    TEXT NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS matches (
    match_id            INTEGER PRIMARY KEY,
    date                TEXT NOT NULL,
    tournament          TEXT NOT NULL,
    home_team_id        INTEGER NOT NULL REFERENCES teams(team_id),
    away_team_id        INTEGER NOT NULL REFERENCES teams(team_id),
    home_goals          INTEGER NOT NULL,
    away_goals          INTEGER NOT NULL,
    city                TEXT,
    country             TEXT,
    neutral             INTEGER NOT NULL,
    shootout_winner_id  INTEGER REFERENCES teams(team_id),
    UNIQUE (date, home_team_id, away_team_id)
);
CREATE TABLE IF NOT EXISTS elo_history (
    team_id  INTEGER NOT NULL REFERENCES teams(team_id),
    date     TEXT NOT NULL,
    match_id INTEGER REFERENCES matches(match_id),
    elo      REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS fifa_rankings (
    team_id      INTEGER REFERENCES teams(team_id),
    team_name    TEXT NOT NULL,
    ranking_date TEXT NOT NULL,
    rank         INTEGER NOT NULL,
    points       REAL
);
"""


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    con.executescript(SCHEMA)
    return con


def load_matches(matches: pd.DataFrame, shootouts: pd.DataFrame,
                 db_path: Path = DB_PATH,
                 processed_dir: Path = PROCESSED_DIR) -> int:
    """Reconstruye teams y matches desde cero (carga idempotente)."""
    con = _connect(db_path)
    try:
        con.execute("DELETE FROM matches")
        con.execute("DELETE FROM teams")

        names = sorted(set(matches["home_team"]) | set(matches["away_team"])
                       | set(shootouts["winner"]))
        teams = pd.DataFrame({"team_id": range(1, len(names) + 1), "name": names})
        teams.to_sql("teams", con, if_exists="append", index=False)
        tid = dict(zip(teams["name"], teams["team_id"]))

        df = matches.copy()
        df["home_team_id"] = df["home_team"].map(tid)
        df["away_team_id"] = df["away_team"].map(tid)
        key = ["date", "home_team", "away_team"]
        so = shootouts.set_index(key)["winner"]
        df["shootout_winner_id"] = (
            df.set_index(key).index.map(so).map(tid).astype("Int64").to_numpy()
        )
        df["neutral"] = df["neutral"].astype(int)
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        out = df[["date", "tournament", "home_team_id", "away_team_id",
                  "home_score", "away_score", "city", "country", "neutral",
                  "shootout_winner_id"]].rename(
            columns={"home_score": "home_goals", "away_score": "away_goals"})
        out.to_sql("matches", con, if_exists="append", index=False)
        con.commit()

        processed_dir.mkdir(parents=True, exist_ok=True)
        matches.to_parquet(processed_dir / "matches.parquet", index=False)
        teams.to_parquet(processed_dir / "teams.parquet", index=False)
        return len(out)
    finally:
        con.close()
