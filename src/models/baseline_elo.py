"""Baseline: regresion logistica multinomial sobre elo_diff -> P(1X2)."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


class EloBaseline:
    def __init__(self):
        self.model_ = LogisticRegression(max_iter=1000)

    def fit(self, df: pd.DataFrame) -> "EloBaseline":
        X = df[["elo_diff"]].to_numpy()
        y = df["outcome"].to_numpy()
        self.model_.fit(X, y)
        return self

    def predict_proba(self, elo_diff: np.ndarray) -> np.ndarray:
        X = np.asarray(elo_diff, dtype=float).reshape(-1, 1)
        proba = self.model_.predict_proba(X)
        # asegurar 3 columnas en orden 0,1,2
        full = np.zeros((len(X), 3))
        for k, cls in enumerate(self.model_.classes_):
            full[:, int(cls)] = proba[:, k]
        return full
