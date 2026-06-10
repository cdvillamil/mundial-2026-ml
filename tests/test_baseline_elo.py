import numpy as np
import pandas as pd

from src.models.baseline_elo import EloBaseline


def _data():
    rng = np.random.default_rng(1)
    rows = []
    for _ in range(200):
        diff = rng.normal(0, 200)
        # mayor diff -> mas probable home win
        p = 1 / (1 + 10 ** (-diff / 400))
        r = rng.random()
        outcome = 0 if r < p * 0.8 else (1 if r < p * 0.8 + 0.2 else 2)
        rows.append({"elo_diff": diff, "outcome": outcome})
    return pd.DataFrame(rows)


def test_elo_baseline_fits_and_predicts():
    model = EloBaseline().fit(_data())
    probs = model.predict_proba(np.array([300.0, -300.0]))
    assert probs.shape == (2, 3)
    assert np.allclose(probs.sum(axis=1), 1.0)
    # con elo_diff alto, home win mas probable que away win
    assert probs[0, 0] > probs[0, 2]
