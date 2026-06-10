"""Features de contexto: descanso, neutral, ventaja local."""
import pandas as pd


def compute_context_features(matches: pd.DataFrame, team_col: str) -> pd.DataFrame:
    """Calcula is_neutral (1=neutral, 0=local/visitante), days_since_last."""
    matches = matches.sort_values("date").reset_index(drop=True)
    rows = []
    team_last_date = {}

    for idx, match in matches.iterrows():
        team = match[team_col]
        row = {"match_idx": idx}

        # is_neutral
        row["is_neutral"] = int(match["neutral"])

        # days_since_last
        if team in team_last_date:
            row["days_since_last"] = (match["date"] - team_last_date[team]).days
        else:
            row["days_since_last"] = None

        team_last_date[team] = match["date"]
        rows.append(row)

    return pd.DataFrame(rows)
