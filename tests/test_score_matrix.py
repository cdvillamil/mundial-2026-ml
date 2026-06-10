import numpy as np

from src.models.score_matrix import most_likely_score, outcome_probs, poisson_score_matrix


def test_matrix_sums_to_one():
    m = poisson_score_matrix(1.5, 1.2, max_goals=10)
    assert abs(m.sum() - 1.0) < 1e-6
    assert m.shape == (11, 11)


def test_outcome_probs_sum_to_one():
    m = poisson_score_matrix(1.5, 1.2)
    ph, pd_, pa = outcome_probs(m)
    assert abs(ph + pd_ + pa - 1.0) < 1e-6
    assert ph > pa  # local con mayor lambda gana mas seguido


def test_equal_lambdas_symmetric():
    m = poisson_score_matrix(1.3, 1.3)
    ph, pd_, pa = outcome_probs(m)
    assert abs(ph - pa) < 1e-6


def test_most_likely_score_returns_tuple():
    m = poisson_score_matrix(2.0, 0.5)
    i, j = most_likely_score(m)
    assert i >= j  # local favorito marca mas
