"""Muestreo de marcadores desde una matriz de probabilidades 11x11."""
import numpy as np


def sample_score(matrix: np.ndarray, rng: np.random.Generator) -> tuple[int, int]:
    """Muestrea (goles_local, goles_visitante) de la matriz."""
    flat = matrix.ravel()
    k = rng.choice(flat.size, p=flat)
    n = matrix.shape[1]
    return int(k // n), int(k % n)


def _penalty_winner(rng: np.random.Generator, elo_diff: float) -> int:
    """0=local gana penales, 1=visitante. Probabilidad logistica suave por Elo."""
    p_home = 1.0 / (1.0 + 10 ** (-elo_diff / 1000.0))  # suave: 400 Elo -> ~72%
    return 0 if rng.random() < p_home else 1


def sample_knockout(matrix: np.ndarray, rng: np.random.Generator,
                    elo_diff: float = 0.0) -> int:
    """Devuelve quien avanza: 0=local, 1=visitante. Empate -> penales (por Elo)."""
    i, j = sample_score(matrix, rng)
    if i > j:
        return 0
    if j > i:
        return 1
    return _penalty_winner(rng, elo_diff)
