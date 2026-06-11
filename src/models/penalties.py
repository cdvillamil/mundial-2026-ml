"""Modelo de tanda de penales: P(gana local) ~ logistica(elo_diff)."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


class PenaltyModel:
    def __init__(self):
        self.model_ = LogisticRegression()
        self._fallback = False

    def fit(self, df: pd.DataFrame) -> "PenaltyModel":
        X = df[["elo_diff"]].to_numpy()
        y = df["home_won"].to_numpy()
        if len(np.unique(y)) < 2:
            self._fallback = True
        else:
            self.model_.fit(X, y)
        return self

    def predict(self, elo_diff: float) -> float:
        if self._fallback:
            return 0.5
        p = self.model_.predict_proba([[elo_diff]])[0]
        idx = list(self.model_.classes_).index(1)  # clase 1 = local gana
        return float(p[idx])


def build_from_db(con) -> "PenaltyModel":
    """Construye el modelo desde los partidos con tanda de penales en la DB."""
    df = pd.read_sql(
        """SELECT m.date, m.home_team_id, m.away_team_id, m.shootout_winner_id
           FROM matches m WHERE m.shootout_winner_id IS NOT NULL""", con,
        parse_dates=["date"])
    if len(df) == 0:
        return PenaltyModel().fit(pd.DataFrame({"elo_diff": [0, 1], "home_won": [0, 1]}))
    elo = pd.read_sql("SELECT team_id, date, elo FROM elo_history", con,
                      parse_dates=["date"]).sort_values("date")
    rows = []
    for r in df.itertuples(index=False):
        eh = elo[(elo["team_id"] == r.home_team_id) & (elo["date"] < r.date)]["elo"]
        ea = elo[(elo["team_id"] == r.away_team_id) & (elo["date"] < r.date)]["elo"]
        if len(eh) == 0 or len(ea) == 0:
            continue
        rows.append({"elo_diff": eh.iloc[-1] - ea.iloc[-1],
                     "home_won": int(r.shootout_winner_id == r.home_team_id)})
    return PenaltyModel().fit(pd.DataFrame(rows))
