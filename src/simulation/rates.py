"""Proveedor de matrices de marcador por emparejamiento (cacheado)."""
import numpy as np

from src.models.gbm_poisson import GBMPoissonModel


class RateProvider:
    """Precomputa/cachea la matriz de marcadores para cada par de equipos
    (sede neutral, importancia = mundial) usando el GBM-Poisson + Elo."""

    def __init__(self, model: GBMPoissonModel, elos: dict[str, float],
                 tournament_importance: int = 3, penalty_model=None):
        self.model = model
        self.elos = elos
        self.imp = tournament_importance
        self.penalty_model = penalty_model
        self._cache: dict[tuple[str, str], np.ndarray] = {}
        self._cumsum: dict[tuple[str, str], tuple[np.ndarray, int]] = {}

    def matrix(self, home: str, away: str) -> np.ndarray:
        key = (home, away)
        if key not in self._cache:
            self._cache[key] = self.model.predict_score_matrix(
                self.elos[home], self.elos[away], self.imp, neutral=True)
        return self._cache[key]

    def cumsum(self, home: str, away: str) -> tuple[np.ndarray, int]:
        """Cumsum aplanado (cacheado) para muestreo rapido del emparejamiento."""
        from src.simulation.match import build_cumsum
        key = (home, away)
        if key not in self._cumsum:
            self._cumsum[key] = build_cumsum(self.matrix(home, away))
        return self._cumsum[key]

    def elo_diff(self, home: str, away: str) -> float:
        return self.elos[home] - self.elos[away]

    def penalty_p_home(self, home: str, away: str):
        """P(local gana penales) si hay modelo de penales; si no, None (fallback Elo)."""
        if self.penalty_model is None:
            return None
        return self.penalty_model.predict(self.elo_diff(home, away))
