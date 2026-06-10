"""Construccion de la matriz de probabilidades de marcador y derivados."""
import numpy as np
from scipy.stats import poisson


def poisson_score_matrix(lambda_home: float, lambda_away: float,
                         max_goals: int = 10) -> np.ndarray:
    """Matriz (max_goals+1)x(max_goals+1) con P(home=i, away=j) bajo Poisson independiente."""
    home = poisson.pmf(np.arange(max_goals + 1), lambda_home)
    away = poisson.pmf(np.arange(max_goals + 1), lambda_away)
    m = np.outer(home, away)
    return m / m.sum()  # renormaliza (cola truncada)


def outcome_probs(matrix: np.ndarray) -> tuple[float, float, float]:
    """(P_home_win, P_draw, P_away_win) desde la matriz."""
    p_home = np.tril(matrix, -1).sum()  # i > j
    p_draw = np.trace(matrix)           # i == j
    p_away = np.triu(matrix, 1).sum()   # i < j
    return float(p_home), float(p_draw), float(p_away)


def most_likely_score(matrix: np.ndarray) -> tuple[int, int]:
    """Marcador (i, j) con mayor probabilidad."""
    i, j = np.unravel_index(np.argmax(matrix), matrix.shape)
    return int(i), int(j)
