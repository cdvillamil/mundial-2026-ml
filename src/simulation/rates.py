"""Proveedor de matrices de marcador por emparejamiento (cacheado)."""
import numpy as np

from src.models.gbm_poisson import GBMPoissonModel


class RateProvider:
    """Precomputa/cachea la matriz de marcadores para cada par de equipos
    (sede neutral, importancia = mundial) usando el GBM-Poisson + Elo."""

    def __init__(self, model: GBMPoissonModel, elos: dict[str, float],
                 tournament_importance: int = 3):
        self.model = model
        self.elos = elos
        self.imp = tournament_importance
        self._cache: dict[tuple[str, str], np.ndarray] = {}

    def matrix(self, home: str, away: str) -> np.ndarray:
        key = (home, away)
        if key not in self._cache:
            self._cache[key] = self.model.predict_score_matrix(
                self.elos[home], self.elos[away], self.imp, neutral=True)
        return self._cache[key]

    def elo_diff(self, home: str, away: str) -> float:
        return self.elos[home] - self.elos[away]
