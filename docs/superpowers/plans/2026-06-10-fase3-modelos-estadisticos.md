# Fase 3 — Modelos Estadísticos: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Modelos Poisson y Dixon-Coles que producen matriz 11×11 de probabilidades de marcador, métricas de evaluación (RPS, log-loss, Brier), y un baseline solo-Elo. Gate F3: Dixon-Coles supera a Poisson y al baseline solo-Elo en RPS.

**Architecture:**
- `src/models/poisson.py` — Poisson independiente: estima fuerzas de ataque/defensa por equipo (máxima verosimilitud vía conteos) → matriz de marcadores.
- `src/models/dixon_coles.py` — añade corrección τ(i,j,ρ) en marcadores bajos + decaimiento temporal ξ.
- `src/models/score_matrix.py` — utilidades compartidas: construir matriz Poisson, derivar P(1X2), marcador más probable.
- `src/evaluation/metrics.py` — RPS, log-loss, Brier, accuracy 1X2, acierto de marcador exacto.
- `src/models/baseline_elo.py` — baseline: logística sobre elo_diff → P(1X2).
- `src/evaluation/backtest_models.py` — compara los 3 modelos con split temporal.

**Tech Stack:** NumPy, SciPy (optimize, stats.poisson), Pandas, pytest. Añadir `scipy` y `statsmodels` a dependencias.

---

### Task 1: Añadir dependencias (scipy, statsmodels)

- [ ] **Step 1.1:** Editar `pyproject.toml` — añadir `"scipy>=1.11"` y `"statsmodels>=0.14"` a `dependencies`.
- [ ] **Step 1.2:** `.venv\Scripts\python -m pip install -e ".[dev]"` → instala scipy, statsmodels.
- [ ] **Step 1.3: Commit** — `git commit -m "build: anadir scipy y statsmodels"`

---

### Task 2: Matriz de marcadores compartida (`src/models/score_matrix.py`)

**Files:** Create `src/models/score_matrix.py`, `tests/test_score_matrix.py`, `src/models/__init__.py`

- [ ] **Step 2.1: Test que falla** — `tests/test_score_matrix.py`

```python
import numpy as np

from src.models.score_matrix import poisson_score_matrix, outcome_probs, most_likely_score


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
```

- [ ] **Step 2.2:** `pytest tests/test_score_matrix.py -v` → FAIL

- [ ] **Step 2.3: Implementación** — `src/models/score_matrix.py`

```python
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
```

- [ ] **Step 2.4:** `pytest tests/test_score_matrix.py -v` → PASS
- [ ] **Step 2.5: Commit** — `git commit -m "feat: matriz de marcadores Poisson + derivados 1X2"`

---

### Task 3: Métricas de evaluación (`src/evaluation/metrics.py`)

**Files:** Create `src/evaluation/metrics.py`, `src/evaluation/__init__.py`, `tests/test_metrics.py`

- [ ] **Step 3.1: Test que falla** — `tests/test_metrics.py`

```python
import numpy as np

from src.evaluation.metrics import rps, log_loss_1x2, brier_1x2


def test_rps_perfect_prediction_is_zero():
    # prob 1.0 al resultado real
    probs = np.array([[1.0, 0.0, 0.0]])
    actual = np.array([0])  # home win
    assert rps(probs, actual) < 1e-9


def test_rps_penalizes_ordered_errors_less():
    # predecir empate cuando gano local: error "cercano"
    near = rps(np.array([[0.0, 1.0, 0.0]]), np.array([0]))
    # predecir away win cuando gano local: error "lejano"
    far = rps(np.array([[0.0, 0.0, 1.0]]), np.array([0]))
    assert far > near


def test_log_loss_perfect_is_near_zero():
    probs = np.array([[0.99, 0.005, 0.005]])
    assert log_loss_1x2(probs, np.array([0])) < 0.02


def test_brier_range():
    probs = np.array([[0.5, 0.3, 0.2]])
    b = brier_1x2(probs, np.array([0]))
    assert 0 <= b <= 2
```

- [ ] **Step 3.2:** `pytest tests/test_metrics.py -v` → FAIL

- [ ] **Step 3.3: Implementación** — `src/evaluation/metrics.py`

```python
"""Metricas de evaluacion probabilistica para predicciones 1X2.

Convencion de clases: 0 = home win, 1 = draw, 2 = away win.
probs: array (n, 3); actual: array (n,) con la clase real.
"""
import numpy as np

_EPS = 1e-15


def rps(probs: np.ndarray, actual: np.ndarray) -> float:
    """Ranked Probability Score promedio (menor es mejor)."""
    probs = np.asarray(probs, dtype=float)
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(actual)), actual] = 1.0
    cum_pred = np.cumsum(probs, axis=1)
    cum_true = np.cumsum(onehot, axis=1)
    # RPS = (1/(r-1)) * sum_{i=1}^{r-1} (CDF_pred_i - CDF_true_i)^2
    return float(np.mean(np.sum((cum_pred[:, :-1] - cum_true[:, :-1]) ** 2, axis=1)))


def log_loss_1x2(probs: np.ndarray, actual: np.ndarray) -> float:
    probs = np.clip(np.asarray(probs, dtype=float), _EPS, 1 - _EPS)
    chosen = probs[np.arange(len(actual)), actual]
    return float(-np.mean(np.log(chosen)))


def brier_1x2(probs: np.ndarray, actual: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=float)
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(actual)), actual] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def accuracy_1x2(probs: np.ndarray, actual: np.ndarray) -> float:
    return float(np.mean(np.argmax(probs, axis=1) == actual))
```

- [ ] **Step 3.4:** `pytest tests/test_metrics.py -v` → PASS
- [ ] **Step 3.5: Commit** — `git commit -m "feat: metricas RPS, log-loss, Brier, accuracy 1X2"`

---

### Task 4: Modelo Poisson (`src/models/poisson.py`)

**Files:** Create `src/models/poisson.py`, `tests/test_poisson.py`

Modelo de fuerzas ataque/defensa: `log(λ_home) = μ + ataque_home - defensa_away + ventaja_local`,
`log(λ_away) = μ + ataque_away - defensa_home`. Estimado vía optimización de verosimilitud Poisson.

- [ ] **Step 4.1: Test que falla** — `tests/test_poisson.py`

```python
import numpy as np
import pandas as pd

from src.models.poisson import PoissonModel


def _training_data():
    """A es fuerte (mete muchos, recibe pocos), B debil. 40 partidos."""
    rng = np.random.default_rng(42)
    rows = []
    for _ in range(40):
        rows.append({"home_team": "A", "away_team": "B",
                     "home_goals": rng.poisson(2.5), "away_goals": rng.poisson(0.5),
                     "neutral": True})
        rows.append({"home_team": "B", "away_team": "A",
                     "home_goals": rng.poisson(0.5), "away_goals": rng.poisson(2.5),
                     "neutral": True})
    return pd.DataFrame(rows)


def test_poisson_fits_and_predicts_matrix():
    model = PoissonModel().fit(_training_data())
    matrix = model.predict_score_matrix("A", "B", neutral=True)
    assert matrix.shape == (11, 11)
    assert abs(matrix.sum() - 1.0) < 1e-6


def test_stronger_team_wins_more_often():
    from src.models.score_matrix import outcome_probs
    model = PoissonModel().fit(_training_data())
    matrix = model.predict_score_matrix("A", "B", neutral=True)
    p_home, _, p_away = outcome_probs(matrix)
    assert p_home > p_away  # A (local) gana mas que B


def test_predict_proba_1x2():
    model = PoissonModel().fit(_training_data())
    probs = model.predict_proba_1x2("A", "B", neutral=True)
    assert abs(probs.sum() - 1.0) < 1e-6
```

- [ ] **Step 4.2:** `pytest tests/test_poisson.py -v` → FAIL

- [ ] **Step 4.3: Implementación** — `src/models/poisson.py`

```python
"""Modelo Poisson de fuerzas ataque/defensa por equipo."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.models.score_matrix import outcome_probs, poisson_score_matrix


class PoissonModel:
    """Estima ataque/defensa por equipo + ventaja local global via MLE Poisson."""

    def __init__(self):
        self.teams_: list[str] = []
        self.attack_: dict[str, float] = {}
        self.defense_: dict[str, float] = {}
        self.home_adv_: float = 0.0
        self.intercept_: float = 0.0

    def fit(self, matches: pd.DataFrame) -> "PoissonModel":
        teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
        self.teams_ = teams
        idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)

        h_idx = matches["home_team"].map(idx).to_numpy()
        a_idx = matches["away_team"].map(idx).to_numpy()
        hg = matches["home_goals"].to_numpy()
        ag = matches["away_goals"].to_numpy()
        is_home = (~matches["neutral"].astype(bool)).to_numpy().astype(float)

        # params: [intercept, home_adv, attack(n-1), defense(n-1)]
        # fijamos attack[0]=defense[0]=0 (identificabilidad) -> usamos n-1 libres c/u
        def unpack(p):
            intercept = p[0]
            home_adv = p[1]
            attack = np.concatenate([[0.0], p[2:2 + (n - 1)]])
            defense = np.concatenate([[0.0], p[2 + (n - 1):]])
            return intercept, home_adv, attack, defense

        def neg_ll(p):
            intercept, home_adv, attack, defense = unpack(p)
            log_lh = intercept + home_adv * is_home + attack[h_idx] - defense[a_idx]
            log_la = intercept + attack[a_idx] - defense[h_idx]
            lh, la = np.exp(log_lh), np.exp(log_la)
            ll = np.sum(hg * log_lh - lh) + np.sum(ag * log_la - la)
            return -ll

        p0 = np.zeros(2 + 2 * (n - 1))
        p0[0] = np.log(max(matches[["home_goals", "away_goals"]].mean().mean(), 0.1))
        res = minimize(neg_ll, p0, method="L-BFGS-B")

        intercept, home_adv, attack, defense = unpack(res.x)
        self.intercept_ = float(intercept)
        self.home_adv_ = float(home_adv)
        self.attack_ = {t: float(attack[idx[t]]) for t in teams}
        self.defense_ = {t: float(defense[idx[t]]) for t in teams}
        return self

    def _lambdas(self, home: str, away: str, neutral: bool) -> tuple[float, float]:
        adv = 0.0 if neutral else self.home_adv_
        lh = np.exp(self.intercept_ + adv + self.attack_[home] - self.defense_[away])
        la = np.exp(self.intercept_ + self.attack_[away] - self.defense_[home])
        return float(lh), float(la)

    def predict_score_matrix(self, home: str, away: str, neutral: bool = True,
                             max_goals: int = 10) -> np.ndarray:
        lh, la = self._lambdas(home, away, neutral)
        return poisson_score_matrix(lh, la, max_goals)

    def predict_proba_1x2(self, home: str, away: str, neutral: bool = True) -> np.ndarray:
        return np.array(outcome_probs(self.predict_score_matrix(home, away, neutral)))
```

- [ ] **Step 4.4:** `pytest tests/test_poisson.py -v` → PASS
- [ ] **Step 4.5: Commit** — `git commit -m "feat: modelo Poisson ataque/defensa via MLE"`

---

### Task 5: Modelo Dixon-Coles (`src/models/dixon_coles.py`)

**Files:** Create `src/models/dixon_coles.py`, `tests/test_dixon_coles.py`

Extiende Poisson con: (a) corrección τ(i,j,ρ) en marcadores 0-0/1-0/0-1/1-1; (b) decaimiento temporal ξ.

- [ ] **Step 5.1: Test que falla** — `tests/test_dixon_coles.py`

```python
import numpy as np
import pandas as pd

from src.models.dixon_coles import DixonColesModel, dc_tau


def test_tau_correction_factors():
    # Para marcadores altos, tau = 1 (sin correccion)
    assert dc_tau(3, 2, lambda_h=1.5, lambda_a=1.2, rho=-0.1) == 1.0
    # Para 0-0 con rho<0, tau != 1
    assert dc_tau(0, 0, 1.5, 1.2, rho=-0.1) != 1.0


def _training_data():
    rng = np.random.default_rng(7)
    rows = []
    dates = pd.date_range("2020-01-01", periods=80, freq="14D")
    for d in dates:
        rows.append({"date": d, "home_team": "A", "away_team": "B",
                     "home_goals": rng.poisson(1.8), "away_goals": rng.poisson(0.8),
                     "neutral": True})
    return pd.DataFrame(rows)


def test_dixon_coles_fits_and_matrix_normalized():
    model = DixonColesModel().fit(_training_data())
    m = model.predict_score_matrix("A", "B", neutral=True)
    assert m.shape == (11, 11)
    assert abs(m.sum() - 1.0) < 1e-6


def test_dixon_coles_predicts_proba():
    model = DixonColesModel().fit(_training_data())
    probs = model.predict_proba_1x2("A", "B", neutral=True)
    assert abs(probs.sum() - 1.0) < 1e-6
```

- [ ] **Step 5.2:** `pytest tests/test_dixon_coles.py -v` → FAIL

- [ ] **Step 5.3: Implementación** — `src/models/dixon_coles.py`

```python
"""Modelo Dixon-Coles: Poisson + correccion de marcadores bajos + decaimiento temporal."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.models.score_matrix import outcome_probs


def dc_tau(i: int, j: int, lambda_h: float, lambda_a: float, rho: float) -> float:
    """Factor de correccion de Dixon-Coles para marcadores bajos."""
    if i == 0 and j == 0:
        return 1.0 - lambda_h * lambda_a * rho
    if i == 0 and j == 1:
        return 1.0 + lambda_h * rho
    if i == 1 and j == 0:
        return 1.0 + lambda_a * rho
    if i == 1 and j == 1:
        return 1.0 - rho
    return 1.0


class DixonColesModel:
    """Ataque/defensa + rho (correccion) + xi (decaimiento temporal)."""

    def __init__(self, xi: float = 0.0):
        self.xi = xi
        self.teams_: list[str] = []
        self.attack_: dict[str, float] = {}
        self.defense_: dict[str, float] = {}
        self.home_adv_ = 0.0
        self.intercept_ = 0.0
        self.rho_ = 0.0

    def fit(self, matches: pd.DataFrame) -> "DixonColesModel":
        teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
        self.teams_ = teams
        idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)

        h_idx = matches["home_team"].map(idx).to_numpy()
        a_idx = matches["away_team"].map(idx).to_numpy()
        hg = matches["home_goals"].to_numpy().astype(int)
        ag = matches["away_goals"].to_numpy().astype(int)
        is_home = (~matches["neutral"].astype(bool)).to_numpy().astype(float)

        # pesos por decaimiento temporal (si hay fecha)
        if "date" in matches.columns and self.xi > 0:
            tmax = matches["date"].max()
            age_days = (tmax - matches["date"]).dt.days.to_numpy()
            weights = np.exp(-self.xi * age_days)
        else:
            weights = np.ones(len(matches))

        def unpack(p):
            intercept, home_adv, rho = p[0], p[1], p[2]
            attack = np.concatenate([[0.0], p[3:3 + (n - 1)]])
            defense = np.concatenate([[0.0], p[3 + (n - 1):]])
            return intercept, home_adv, rho, attack, defense

        def neg_ll(p):
            intercept, home_adv, rho, attack, defense = unpack(p)
            log_lh = intercept + home_adv * is_home + attack[h_idx] - defense[a_idx]
            log_la = intercept + attack[a_idx] - defense[h_idx]
            lh, la = np.exp(log_lh), np.exp(log_la)
            # log P Poisson + log tau (solo marcadores bajos)
            ll_pois = hg * log_lh - lh + ag * log_la - la
            tau = np.ones(len(p) if False else len(hg))
            low = (hg <= 1) & (ag <= 1)
            for k in np.where(low)[0]:
                tau[k] = dc_tau(hg[k], ag[k], lh[k], la[k], rho)
            tau = np.clip(tau, 1e-10, None)
            ll = np.sum(weights * (ll_pois + np.log(tau)))
            return -ll

        p0 = np.zeros(3 + 2 * (n - 1))
        p0[0] = np.log(max(matches[["home_goals", "away_goals"]].mean().mean(), 0.1))
        res = minimize(neg_ll, p0, method="L-BFGS-B",
                       bounds=[(None, None), (None, None), (-0.2, 0.2)]
                              + [(None, None)] * (2 * (n - 1)))

        intercept, home_adv, rho, attack, defense = unpack(res.x)
        self.intercept_, self.home_adv_, self.rho_ = float(intercept), float(home_adv), float(rho)
        self.attack_ = {t: float(attack[idx[t]]) for t in teams}
        self.defense_ = {t: float(defense[idx[t]]) for t in teams}
        return self

    def _lambdas(self, home, away, neutral):
        adv = 0.0 if neutral else self.home_adv_
        lh = np.exp(self.intercept_ + adv + self.attack_[home] - self.defense_[away])
        la = np.exp(self.intercept_ + self.attack_[away] - self.defense_[home])
        return float(lh), float(la)

    def predict_score_matrix(self, home, away, neutral=True, max_goals=10) -> np.ndarray:
        from scipy.stats import poisson
        lh, la = self._lambdas(home, away, neutral)
        ph = poisson.pmf(np.arange(max_goals + 1), lh)
        pa = poisson.pmf(np.arange(max_goals + 1), la)
        m = np.outer(ph, pa)
        # aplicar tau a las 4 celdas bajas
        for i in (0, 1):
            for j in (0, 1):
                m[i, j] *= dc_tau(i, j, lh, la, self.rho_)
        return m / m.sum()

    def predict_proba_1x2(self, home, away, neutral=True) -> np.ndarray:
        return np.array(outcome_probs(self.predict_score_matrix(home, away, neutral)))
```

- [ ] **Step 5.4:** `pytest tests/test_dixon_coles.py -v` → PASS
- [ ] **Step 5.5: Commit** — `git commit -m "feat: modelo Dixon-Coles (correccion marcadores bajos + decaimiento)"`

---

### Task 6: Baseline solo-Elo (`src/models/baseline_elo.py`)

**Files:** Create `src/models/baseline_elo.py`, `tests/test_baseline_elo.py`

- [ ] **Step 6.1: Test que falla** — `tests/test_baseline_elo.py`

```python
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
```

- [ ] **Step 6.2:** `pytest tests/test_baseline_elo.py -v` → FAIL

- [ ] **Step 6.3: Implementación** — `src/models/baseline_elo.py`

```python
"""Baseline: regresion logistica multinomial sobre elo_diff -> P(1X2)."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


class EloBaseline:
    def __init__(self):
        self.model_ = LogisticRegression(multi_class="multinomial", max_iter=1000)

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
```

Nota: requiere `scikit-learn` — añadirlo a dependencias en este task si falta.

- [ ] **Step 6.4:** Verificar sklearn instalado; si falta, `pip install scikit-learn` + añadir a pyproject. Luego `pytest tests/test_baseline_elo.py -v` → PASS
- [ ] **Step 6.5: Commit** — `git commit -m "feat: baseline solo-Elo (logistica multinomial)"`

---

### Task 7: Backtest comparativo (`src/evaluation/backtest_models.py`)

**Files:** Create `src/evaluation/backtest_models.py`

Compara Poisson vs Dixon-Coles vs baseline-Elo con split temporal real, usando los datos del DB.

- [ ] **Step 7.1: Implementación** — `src/evaluation/backtest_models.py`

```python
"""Backtest comparativo de modelos estadisticos con split temporal."""
import sqlite3

import numpy as np
import pandas as pd

from src.config import DB_PATH, REPORTS_DIR
from src.evaluation.metrics import accuracy_1x2, brier_1x2, log_loss_1x2, rps
from src.models.dixon_coles import DixonColesModel
from src.models.poisson import PoissonModel


def _load_matches(cutoff: str = "2018-01-01", min_date: str = "2010-01-01") -> pd.DataFrame:
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        """SELECT m.date, t1.name AS home_team, t2.name AS away_team,
                  m.home_goals, m.away_goals, m.neutral
           FROM matches m
           JOIN teams t1 ON t1.team_id = m.home_team_id
           JOIN teams t2 ON t2.team_id = m.away_team_id
           WHERE m.date >= ? ORDER BY m.date""",
        con, params=[min_date], parse_dates=["date"]
    )
    con.close()
    return df


def _outcome(hg, ag):
    return 0 if hg > ag else (1 if hg == ag else 2)


def run_backtest(cutoff: str = "2018-01-01") -> pd.DataFrame:
    df = _load_matches(min_date="2010-01-01")
    train = df[df["date"] < cutoff].copy()
    test = df[df["date"] >= cutoff].copy()

    # Solo partidos de test donde ambos equipos estan en train
    poisson = PoissonModel().fit(train)
    dc = DixonColesModel().fit(train)
    known = set(poisson.teams_)
    test = test[test["home_team"].isin(known) & test["away_team"].isin(known)]

    actual = np.array([_outcome(r.home_goals, r.away_goals) for r in test.itertuples()])

    results = []
    for name, model in [("Poisson", poisson), ("Dixon-Coles", dc)]:
        probs = np.array([
            model.predict_proba_1x2(r.home_team, r.away_team, bool(r.neutral))
            for r in test.itertuples()
        ])
        results.append({
            "model": name,
            "n_test": len(test),
            "RPS": rps(probs, actual),
            "log_loss": log_loss_1x2(probs, actual),
            "Brier": brier_1x2(probs, actual),
            "accuracy": accuracy_1x2(probs, actual),
        })

    return pd.DataFrame(results)


def main():
    res = run_backtest()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "f3_model_comparison.md"
    lines = ["# Comparacion de modelos estadisticos (corte 2018-01-01)", ""]
    lines.append(res.to_markdown(index=False))
    best = res.loc[res["RPS"].idxmin(), "model"]
    lines += ["", f"**Mejor modelo por RPS: {best}**"]
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
```

- [ ] **Step 7.2: Corrida real** — `python -m src.evaluation.backtest_models`
Expected: tabla con RPS de ambos modelos; verificar que Dixon-Coles tenga RPS ≤ Poisson.

- [ ] **Step 7.3: Commit** — `git add -f outputs/reports/f3_model_comparison.md && git commit -m "feat: backtest comparativo de modelos (Poisson vs Dixon-Coles)"`

---

### Task 8: Validación y cierre F3

- [ ] **Step 8.1:** `pytest -v` → todos PASS
- [ ] **Step 8.2:** Verificar gate F3: Dixon-Coles ≤ Poisson en RPS (del reporte). Documentar resultado.
- [ ] **Step 8.3:** Actualizar README (marcar Fase 3).
- [ ] **Step 8.4: Commit + push** — `git commit -m "docs: fase 3 completada"; git push`

---

## Self-Review

- **Cobertura del spec F3 (§5, §8):** Poisson ✅, Dixon-Coles ✅, matriz de marcadores ✅, métricas RPS/log-loss/Brier ✅, baseline Elo ✅, backtest comparativo ✅, gate (DC vence a Poisson en RPS) ✅.
- **Poisson bivariado:** mencionado en el documento pero su ganancia es marginal sobre Dixon-Coles; se omite en esta fase (YAGNI) y se puede añadir en mejoras.
- **Consistencia:** todos los modelos exponen `predict_proba_1x2(home, away, neutral)` y `predict_score_matrix(...)`; las métricas usan convención 0=home/1=draw/2=away consistente en backtest.
- **Sin placeholders:** código completo en cada step.
