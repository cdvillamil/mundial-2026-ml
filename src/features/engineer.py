"""Pipeline de feature engineering: lee matches, genera features, escribe matriz."""
import sqlite3
from pathlib import Path

import pandas as pd

from src.config import DB_PATH, PROCESSED_DIR


def engineer_features(db_path: Path = DB_PATH, processed_dir: Path = PROCESSED_DIR) -> int:
    """Lee matches del DB, genera matriz de features (estructura base)."""
    con = sqlite3.connect(db_path)
    matches = pd.read_sql(
        """SELECT m.*,
                  t1.name AS home_team,
                  t2.name AS away_team
           FROM matches m
           JOIN teams t1 ON t1.team_id = m.home_team_id
           JOIN teams t2 ON t2.team_id = m.away_team_id
           ORDER BY m.date""",
        con, parse_dates=["date"]
    )
    con.close()

    # Matriz base con estructura de features (schema sin llenarlas aún)
    matrix = matches[["match_id", "date", "home_team", "away_team",
                      "home_goals", "away_goals", "neutral"]].copy()

    # Estructura de features para futuro cálculo (columnas vacías por ahora)
    for window in [5, 10, 20]:
        for team_prefix in ["home_", "away_"]:
            for feat in ["form_wins", "form_draws", "form_losses", "form_gf", "form_ga", "is_neutral", "days_since_last"]:
                matrix[f"{team_prefix}{feat}_{window}"] = None

    processed_dir.mkdir(parents=True, exist_ok=True)
    matrix.to_parquet(processed_dir / "features_matrix.parquet", index=False)
    return len(matrix)


if __name__ == "__main__":
    n = engineer_features()
    print(f"Generated features_matrix with {n} rows")
