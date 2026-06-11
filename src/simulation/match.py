"""Muestreo de marcadores desde una matriz de probabilidades 11x11.

El muestreo rapido usa la suma acumulada (cumsum) + searchsorted. En el bucle
Monte Carlo, RateProvider cachea el cumsum por emparejamiento (sin riesgo de
colision de id), y aqui solo se hace el muestreo.
"""
import numpy as np


def build_cumsum(matrix: np.ndarray) -> tuple[np.ndarray, int]:
    """Devuelve (cumsum_aplanado, n_columnas) listo para muestrear."""
    cumsum = np.cumsum(matrix.ravel())
    cumsum[-1] = 1.0  # robustez ante error de redondeo
    return cumsum, matrix.shape[1]


def sample_from_cumsum(cumsum: np.ndarray, n: int,
                       rng: np.random.Generator) -> tuple[int, int]:
    """Muestrea (goles_local, goles_visitante) desde un cumsum precomputado."""
    k = int(np.searchsorted(cumsum, rng.random()))
    return k // n, k % n


def sample_score(matrix: np.ndarray, rng: np.random.Generator) -> tuple[int, int]:
    """Muestrea (goles_local, goles_visitante) de la matriz."""
    cumsum, n = build_cumsum(matrix)
    return sample_from_cumsum(cumsum, n, rng)


def _penalty_winner(rng: np.random.Generator, elo_diff: float) -> int:
    """0=local gana penales, 1=visitante. Probabilidad logistica suave por Elo."""
    p_home = 1.0 / (1.0 + 10 ** (-elo_diff / 1000.0))  # suave: 400 Elo -> ~72%
    return 0 if rng.random() < p_home else 1


def knockout_from_cumsum(cumsum: np.ndarray, n: int, rng: np.random.Generator,
                         elo_diff: float = 0.0) -> int:
    """Quien avanza (0=local, 1=visitante) desde un cumsum precomputado."""
    i, j = sample_from_cumsum(cumsum, n, rng)
    if i > j:
        return 0
    if j > i:
        return 1
    return _penalty_winner(rng, elo_diff)


def sample_knockout(matrix: np.ndarray, rng: np.random.Generator,
                    elo_diff: float = 0.0) -> int:
    """Devuelve quien avanza: 0=local, 1=visitante. Empate -> penales (por Elo)."""
    cumsum, n = build_cumsum(matrix)
    return knockout_from_cumsum(cumsum, n, rng, elo_diff)
