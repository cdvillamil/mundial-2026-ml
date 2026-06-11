import numpy as np

from src.models.score_matrix import poisson_score_matrix
from src.simulation.match import sample_knockout, sample_score


def test_sample_score_returns_valid_goals():
    m = poisson_score_matrix(1.5, 1.2)
    rng = np.random.default_rng(0)
    i, j = sample_score(m, rng)
    assert 0 <= i <= 10 and 0 <= j <= 10


def test_sample_score_distribution_matches_matrix():
    # lambda local alto -> en promedio local marca mas
    m = poisson_score_matrix(3.0, 0.5)
    rng = np.random.default_rng(1)
    samples = [sample_score(m, rng) for _ in range(2000)]
    avg_home = np.mean([s[0] for s in samples])
    avg_away = np.mean([s[1] for s in samples])
    assert avg_home > avg_away
    assert abs(avg_home - 3.0) < 0.3  # cerca de lambda


def test_knockout_never_returns_draw():
    m = poisson_score_matrix(1.0, 1.0)
    rng = np.random.default_rng(2)
    for _ in range(50):
        winner = sample_knockout(m, rng, elo_diff=0.0)
        assert winner in (0, 1)  # 0=home avanza, 1=away avanza


def test_knockout_favors_stronger_on_penalties():
    # matriz simetrica -> el desempate por penales decide; elo alto favorece
    m = poisson_score_matrix(1.0, 1.0)
    rng = np.random.default_rng(3)
    wins_home = sum(sample_knockout(m, rng, elo_diff=400.0) == 0 for _ in range(400))
    assert wins_home > 200  # ventaja clara del favorito
