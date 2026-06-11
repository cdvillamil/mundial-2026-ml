# Fase 4 — Modelos ML y Ensamble: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Modelo GBM-Poisson que usa features de Elo (point-in-time, anti-leakage) para predecir tasas de gol → matriz de marcadores con corrección Dixon-Coles. Calibración isotónica de 1X2. Backtest que compara GBM vs los baselines de Fase 3. Gate F4: GBM-Poisson mejora el RPS del mejor modelo de Fase 3, y ECE < 0.05.

**Architecture:**
- `src/features/elo_features.py` — para cada partido, busca el Elo de cada equipo **estrictamente anterior** a la fecha (point-in-time, sin leakage), desde la tabla `elo_history`. Produce dataframe con `home_elo`, `away_elo`, `elo_diff`, `tournament_importance`.
- `src/models/gbm_poisson.py` — `HistGradientBoostingRegressor(loss="poisson")` de sklearn (sustituto justificado de LightGBM/XGBoost por Python 3.14). Filas simétricas (una por equipo-partido) → predice λ. Reusa `score_matrix` + corrección Dixon-Coles para la matriz.
- `src/models/calibration.py` — calibración isotónica de probabilidades 1X2 + cálculo de ECE.
- `src/evaluation/backtest_models.py` — extender para incluir GBM-Poisson.

**Decisión técnica documentada:** se usa `sklearn.ensemble.HistGradientBoostingRegressor(loss="poisson")` en lugar de LightGBM/XGBoost. Razón: Python 3.14.5 carece de wheels estables de esas librerías en Windows; sklearn ya es dependencia, soporta Poisson nativo y el principio es idéntico (árboles boosting con objetivo Poisson). Migrar a LightGBM en el futuro es trivial (misma interfaz de features).

**Tech Stack:** scikit-learn (HistGradientBoostingRegressor, IsotonicRegression), NumPy, Pandas, SciPy, pytest.

---

### Task 1: Features de Elo point-in-time (`src/features/elo_features.py`)

**Files:** Create `src/features/elo_features.py`, `tests/test_elo_features.py`

- [ ] **Step 1.1: Test que falla** — `tests/test_elo_features.py`

```python
import pandas as pd

from src.features.elo_features import attach_pre_match_elo, tournament_importance


def test_tournament_importance_levels():
    assert tournament_importance("Friendly") == 0
    assert tournament_importance("FIFA World Cup qualification") == 1
    assert tournament_importance("Copa América") == 2
    assert tournament_importance("FIFA World Cup") == 3


def test_attach_pre_match_elo_is_point_in_time():
    # elo_history: A sube de 1500 a 1600 tras un partido el 2020-01-10
    elo_history = pd.DataFrame({
        "team": ["A", "B", "A"],
        "date": pd.to_datetime(["2020-01-10", "2020-01-10", "2020-06-01"]),
        "elo": [1600.0, 1400.0, 1650.0],
    })
    matches = pd.DataFrame({
        "date": pd.to_datetime(["2020-03-01"]),
        "home_team": ["A"], "away_team": ["B"],
        "tournament": ["Friendly"],
    })
    out = attach_pre_match_elo(matches, elo_history)
    # En 2020-03-01, el ultimo Elo de A previo es 1600 (no 1650, que es de junio)
    assert out.iloc[0]["home_elo"] == 1600.0
    assert out.iloc[0]["away_elo"] == 1400.0
    assert out.iloc[0]["elo_diff"] == 200.0


def test_attach_handles_missing_elo_with_default():
    elo_history = pd.DataFrame({"team": ["A"], "date": pd.to_datetime(["2020-01-10"]), "elo": [1600.0]})
    matches = pd.DataFrame({
        "date": pd.to_datetime(["2020-03-01"]),
        "home_team": ["A"], "away_team": ["NEW"],
        "tournament": ["Friendly"],
    })
    out = attach_pre_match_elo(matches, elo_history)
    assert out.iloc[0]["away_elo"] == 1500.0  # default para equipo sin historia
```

- [ ] **Step 1.2:** `pytest tests/test_elo_features.py -v` → FAIL

- [ ] **Step 1.3: Implementación** — `src/features/elo_features.py`

```python
"""Features de Elo point-in-time (anti-leakage) para cada partido."""
import numpy as np
import pandas as pd

DEFAULT_ELO = 1500.0

_CONTINENTAL = {
    "UEFA Euro", "Copa América", "African Cup of Nations", "AFC Asian Cup",
    "CONCACAF Championship", "Gold Cup", "Oceania Nations Cup", "Confederations Cup",
}


def tournament_importance(tournament: str) -> int:
    """0=amistoso, 1=eliminatoria, 2=copa continental, 3=mundial."""
    if tournament == "FIFA World Cup":
        return 3
    if tournament in _CONTINENTAL:
        return 2
    if "qualification" in tournament.lower():
        return 1
    if tournament == "Friendly":
        return 0
    return 1


def _latest_elo_before(elo_sorted: pd.DataFrame, team: str, date) -> float:
    """Ultimo Elo del equipo estrictamente antes de date (point-in-time)."""
    sub = elo_sorted[(elo_sorted["team"] == team) & (elo_sorted["date"] < date)]
    if len(sub) == 0:
        return DEFAULT_ELO
    return float(sub["elo"].iloc[-1])


def attach_pre_match_elo(matches: pd.DataFrame, elo_history: pd.DataFrame) -> pd.DataFrame:
    """Agrega home_elo, away_elo, elo_diff, tournament_importance a matches."""
    elo_sorted = elo_history.sort_values(["team", "date"]).reset_index(drop=True)
    out = matches.copy().reset_index(drop=True)

    home_elos, away_elos = [], []
    for m in out.itertuples(index=False):
        home_elos.append(_latest_elo_before(elo_sorted, m.home_team, m.date))
        away_elos.append(_latest_elo_before(elo_sorted, m.away_team, m.date))

    out["home_elo"] = home_elos
    out["away_elo"] = away_elos
    out["elo_diff"] = out["home_elo"] - out["away_elo"]
    out["tournament_importance"] = out["tournament"].map(tournament_importance)
    return out
```

- [ ] **Step 1.4:** `pytest tests/test_elo_features.py -v` → PASS
- [ ] **Step 1.5: Commit** — `git commit -m "feat: features de Elo point-in-time (anti-leakage)"`

---

### Task 2: Modelo GBM-Poisson (`src/models/gbm_poisson.py`)

**Files:** Create `src/models/gbm_poisson.py`, `tests/test_gbm_poisson.py`

Filas simétricas: cada partido genera 2 filas (equipo que anota = "self"). Features: `self_elo`, `opp_elo`, `elo_diff_signed`, `is_home`, `tournament_importance`. Target = goles anotados. Un único GBM aprende a predecir λ; en inferencia se llama dos veces (perspectiva local y visitante).

- [ ] **Step 2.1: Test que falla** — `tests/test_gbm_poisson.py`

```python
import numpy as np
import pandas as pd

from src.models.gbm_poisson import GBMPoissonModel


def _training_data():
    """Equipos con Elo alto anotan mas; entrenamiento sintetico coherente."""
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(400):
        he = rng.uniform(1400, 2000)
        ae = rng.uniform(1400, 2000)
        # lambda crece con ventaja de Elo
        lh = np.exp(0.0 + (he - ae) / 600 + 0.2)  # +0.2 ventaja local
        la = np.exp(0.0 + (ae - he) / 600)
        rows.append({
            "home_team": "H", "away_team": "A",
            "home_elo": he, "away_elo": ae, "elo_diff": he - ae,
            "tournament_importance": 1, "neutral": False,
            "home_goals": rng.poisson(lh), "away_goals": rng.poisson(la),
        })
    return pd.DataFrame(rows)


def test_gbm_fits_and_predicts_lambdas():
    model = GBMPoissonModel().fit(_training_data())
    lh, la = model.predict_lambdas(home_elo=1900, away_elo=1500,
                                   tournament_importance=1, neutral=False)
    assert lh > la  # equipo local mucho mas fuerte anota mas
    assert lh > 0 and la > 0


def test_gbm_predicts_score_matrix():
    model = GBMPoissonModel().fit(_training_data())
    m = model.predict_score_matrix(home_elo=1700, away_elo=1700,
                                    tournament_importance=1, neutral=False)
    assert m.shape == (11, 11)
    assert abs(m.sum() - 1.0) < 1e-6


def test_gbm_proba_1x2_sums_to_one():
    model = GBMPoissonModel().fit(_training_data())
    probs = model.predict_proba_1x2(home_elo=1800, away_elo=1500,
                                     tournament_importance=3, neutral=True)
    assert abs(probs.sum() - 1.0) < 1e-6
    assert probs[0] > probs[2]  # local favorito gana mas
```

- [ ] **Step 2.2:** `pytest tests/test_gbm_poisson.py -v` → FAIL

- [ ] **Step 2.3: Implementación** — `src/models/gbm_poisson.py`

```python
"""GBM-Poisson: HistGradientBoostingRegressor(loss=poisson) sobre features de Elo.

Sustituto justificado de LightGBM/XGBoost (Python 3.14 sin wheels estables).
Filas simetricas: cada partido -> 2 filas (perspectiva del equipo que anota).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.models.dixon_coles import dc_tau
from src.models.score_matrix import outcome_probs
from scipy.stats import poisson

_FEATURES = ["self_elo", "opp_elo", "elo_diff_signed", "is_home", "tournament_importance"]


def _build_symmetric_rows(matches: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Convierte cada partido en 2 filas (local anota / visitante anota)."""
    is_home_flag = (~matches["neutral"].astype(bool)).astype(int)
    home_rows = pd.DataFrame({
        "self_elo": matches["home_elo"].to_numpy(),
        "opp_elo": matches["away_elo"].to_numpy(),
        "elo_diff_signed": (matches["home_elo"] - matches["away_elo"]).to_numpy(),
        "is_home": is_home_flag.to_numpy(),
        "tournament_importance": matches["tournament_importance"].to_numpy(),
    })
    away_rows = pd.DataFrame({
        "self_elo": matches["away_elo"].to_numpy(),
        "opp_elo": matches["home_elo"].to_numpy(),
        "elo_diff_signed": (matches["away_elo"] - matches["home_elo"]).to_numpy(),
        "is_home": np.zeros(len(matches), dtype=int),  # visitante nunca tiene ventaja local
        "tournament_importance": matches["tournament_importance"].to_numpy(),
    })
    X = pd.concat([home_rows, away_rows], ignore_index=True)
    y = np.concatenate([matches["home_goals"].to_numpy(), matches["away_goals"].to_numpy()])
    return X, y


class GBMPoissonModel:
    def __init__(self, rho: float = -0.05, max_iter: int = 200, **kwargs):
        self.rho = rho
        self.model_ = HistGradientBoostingRegressor(
            loss="poisson", max_iter=max_iter, learning_rate=0.05,
            max_depth=4, min_samples_leaf=50, **kwargs
        )

    def fit(self, matches: pd.DataFrame) -> "GBMPoissonModel":
        X, y = _build_symmetric_rows(matches)
        self.model_.fit(X[_FEATURES], y)
        return self

    def predict_lambdas(self, home_elo, away_elo, tournament_importance, neutral):
        is_home = 0 if neutral else 1
        row_h = [[home_elo, away_elo, home_elo - away_elo, is_home, tournament_importance]]
        row_a = [[away_elo, home_elo, away_elo - home_elo, 0, tournament_importance]]
        lh = float(self.model_.predict(pd.DataFrame(row_h, columns=_FEATURES))[0])
        la = float(self.model_.predict(pd.DataFrame(row_a, columns=_FEATURES))[0])
        return lh, la

    def predict_score_matrix(self, home_elo, away_elo, tournament_importance,
                             neutral, max_goals=10) -> np.ndarray:
        lh, la = self.predict_lambdas(home_elo, away_elo, tournament_importance, neutral)
        ph = poisson.pmf(np.arange(max_goals + 1), lh)
        pa = poisson.pmf(np.arange(max_goals + 1), la)
        m = np.outer(ph, pa)
        for i in (0, 1):
            for j in (0, 1):
                m[i, j] *= dc_tau(i, j, lh, la, self.rho)
        return m / m.sum()

    def predict_proba_1x2(self, home_elo, away_elo, tournament_importance,
                          neutral) -> np.ndarray:
        return np.array(outcome_probs(self.predict_score_matrix(
            home_elo, away_elo, tournament_importance, neutral)))
```

- [ ] **Step 2.4:** `pytest tests/test_gbm_poisson.py -v` → PASS
- [ ] **Step 2.5: Commit** — `git commit -m "feat: modelo GBM-Poisson (HistGB) + capa Dixon-Coles"`

---

### Task 3: Calibración isotónica + ECE (`src/models/calibration.py`)

**Files:** Create `src/models/calibration.py`, `tests/test_calibration.py`

- [ ] **Step 3.1: Test que falla** — `tests/test_calibration.py`

```python
import numpy as np

from src.models.calibration import expected_calibration_error, IsotonicCalibrator1x2


def test_ece_perfect_is_zero():
    # 100 predicciones de 1.0 al evento real -> ECE ~ 0
    probs = np.tile([1.0, 0.0, 0.0], (100, 1))
    actual = np.zeros(100, dtype=int)
    assert expected_calibration_error(probs, actual) < 1e-6


def test_ece_detects_miscalibration():
    # decir 0.9 pero acertar solo 50%
    probs = np.tile([0.9, 0.05, 0.05], (100, 1))
    actual = np.array([0] * 50 + [2] * 50)
    assert expected_calibration_error(probs, actual) > 0.2


def test_isotonic_calibrator_normalizes():
    rng = np.random.default_rng(0)
    probs = rng.dirichlet([2, 2, 2], size=300)
    actual = rng.integers(0, 3, size=300)
    cal = IsotonicCalibrator1x2().fit(probs, actual)
    out = cal.transform(probs)
    assert np.allclose(out.sum(axis=1), 1.0)
    assert out.shape == probs.shape
```

- [ ] **Step 3.2:** `pytest tests/test_calibration.py -v` → FAIL

- [ ] **Step 3.3: Implementación** — `src/models/calibration.py`

```python
"""Calibracion isotonica de probabilidades 1X2 + ECE."""
import numpy as np
from sklearn.isotonic import IsotonicRegression


def expected_calibration_error(probs: np.ndarray, actual: np.ndarray,
                               n_bins: int = 10) -> float:
    """ECE sobre la clase predicha (confianza vs acierto)."""
    probs = np.asarray(probs, dtype=float)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == actual).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(actual)
    for b in range(n_bins):
        mask = (conf > bins[b]) & (conf <= bins[b + 1])
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


class IsotonicCalibrator1x2:
    """Una regresion isotonica por clase + renormalizacion."""

    def __init__(self):
        self.iso_ = [IsotonicRegression(out_of_bounds="clip") for _ in range(3)]

    def fit(self, probs: np.ndarray, actual: np.ndarray) -> "IsotonicCalibrator1x2":
        for c in range(3):
            target = (actual == c).astype(float)
            self.iso_[c].fit(probs[:, c], target)
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        cal = np.column_stack([self.iso_[c].transform(probs[:, c]) for c in range(3)])
        row_sums = cal.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return cal / row_sums
```

- [ ] **Step 3.4:** `pytest tests/test_calibration.py -v` → PASS
- [ ] **Step 3.5: Commit** — `git commit -m "feat: calibracion isotonica 1X2 + metrica ECE"`

---

### Task 4: Backtest extendido con GBM (`src/evaluation/backtest_models.py`)

**Files:** Modify `src/evaluation/backtest_models.py`

Añadir GBM-Poisson (con features de Elo point-in-time) a la comparación, más reporte de ECE.

- [ ] **Step 4.1: Implementación** — reescribir `run_backtest` para incluir GBM

```python
"""Backtest comparativo de modelos con split temporal."""
import sqlite3

import numpy as np
import pandas as pd

from src.config import DB_PATH, REPORTS_DIR
from src.evaluation.metrics import accuracy_1x2, brier_1x2, log_loss_1x2, rps
from src.features.elo_features import attach_pre_match_elo, tournament_importance
from src.models.calibration import expected_calibration_error
from src.models.dixon_coles import DixonColesModel
from src.models.gbm_poisson import GBMPoissonModel
from src.models.poisson import PoissonModel


def _load(min_date="2010-01-01"):
    con = sqlite3.connect(DB_PATH)
    matches = pd.read_sql(
        """SELECT m.date, t1.name AS home_team, t2.name AS away_team,
                  m.home_goals, m.away_goals, m.neutral, m.tournament
           FROM matches m
           JOIN teams t1 ON t1.team_id = m.home_team_id
           JOIN teams t2 ON t2.team_id = m.away_team_id
           WHERE m.date >= ? ORDER BY m.date""",
        con, params=[min_date], parse_dates=["date"])
    elo = pd.read_sql(
        "SELECT t.name AS team, e.date, e.elo FROM elo_history e "
        "JOIN teams t ON t.team_id = e.team_id",
        con, parse_dates=["date"])
    con.close()
    return matches, elo


def _outcome(hg, ag):
    return 0 if hg > ag else (1 if hg == ag else 2)


def run_backtest(cutoff="2018-01-01"):
    matches, elo = _load()
    matches = attach_pre_match_elo(matches, elo)
    train = matches[matches["date"] < cutoff].copy()
    test = matches[matches["date"] >= cutoff].copy()

    poisson = PoissonModel().fit(train)
    dc = DixonColesModel().fit(train)
    gbm = GBMPoissonModel().fit(train)

    known = set(poisson.teams_)
    test_cls = test[test["home_team"].isin(known) & test["away_team"].isin(known)].copy()
    actual = np.array([_outcome(r.home_goals, r.away_goals) for r in test_cls.itertuples()])

    rows = []
    # Poisson y DC (basados en equipo)
    for name, model in [("Poisson", poisson), ("Dixon-Coles", dc)]:
        probs = np.array([model.predict_proba_1x2(r.home_team, r.away_team, bool(r.neutral))
                          for r in test_cls.itertuples()])
        rows.append((name, probs))

    # GBM (basado en Elo; funciona tambien con equipos no vistos)
    gbm_probs = np.array([
        gbm.predict_proba_1x2(r.home_elo, r.away_elo, r.tournament_importance, bool(r.neutral))
        for r in test_cls.itertuples()])
    rows.append(("GBM-Poisson", gbm_probs))

    results = []
    for name, probs in rows:
        results.append({
            "model": name, "n_test": len(test_cls),
            "RPS": round(rps(probs, actual), 4),
            "log_loss": round(log_loss_1x2(probs, actual), 4),
            "Brier": round(brier_1x2(probs, actual), 4),
            "accuracy": round(accuracy_1x2(probs, actual), 4),
            "ECE": round(expected_calibration_error(probs, actual), 4),
        })
    return pd.DataFrame(results)


def main():
    res = run_backtest()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "f4_model_comparison.md"
    best = res.loc[res["RPS"].idxmin(), "model"]
    lines = [
        "# Comparacion de modelos con GBM (corte temporal 2018-01-01)",
        "", "Entrenamiento: 2010-2018. Test: 2018+. GBM usa Elo point-in-time.",
        "", res.to_markdown(index=False), "",
        f"**Mejor modelo por RPS: {best}**", "",
        "Nota: GBM-Poisson usa HistGradientBoostingRegressor(loss=poisson) de sklearn",
        "(sustituto de LightGBM/XGBoost por Python 3.14). Predice desde Elo, no identidad",
        "del equipo, por lo que generaliza a equipos no vistos en entrenamiento.",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4.2: Corrida real** — `python -m src.evaluation.backtest_models`
Expected: tabla con 3 modelos; verificar RPS de GBM y su ECE.

- [ ] **Step 4.3: Commit** — `git add -f outputs/reports/f4_model_comparison.md && git commit -m "feat: backtest con GBM-Poisson + ECE"`

---

### Task 5: Validación y cierre F4

- [ ] **Step 5.1:** `pytest -v` → todos PASS
- [ ] **Step 5.2:** Evaluar gate F4 contra el reporte: ¿GBM mejora el RPS del mejor de Fase 3 (0.3596)? ¿ECE < 0.05? Documentar el resultado honestamente (si no mejora, explicar por qué y qué se aprende).
- [ ] **Step 5.3:** Actualizar README (marcar Fase 4 con métricas reales).
- [ ] **Step 5.4: Commit + push** — `git commit -m "docs: fase 4 completada"; git push`

---

## Self-Review

- **Cobertura del spec F4 (§5.4, §5.5, §8):** GBM-Poisson ✅, capa Dixon-Coles sobre GBM ✅, calibración isotónica ✅, ECE ✅, backtest comparativo vs baselines ✅, gate (mejora RPS + ECE<0.05) ✅.
- **Desviación documentada:** HistGradientBoostingRegressor en vez de LightGBM/XGBoost (Python 3.14). Interfaz de features idéntica → migración trivial.
- **Ensamble:** el documento menciona ensamble como paso final; con solo features de Elo el ensamble aporta poco, se deja para mejoras (§10). El foco de F4 es el GBM híbrido y la calibración.
- **Anti-leakage:** `attach_pre_match_elo` usa Elo estrictamente anterior a la fecha del partido; test explícito lo verifica.
- **Consistencia:** GBM expone `predict_proba_1x2(...)` y `predict_score_matrix(...)` como los modelos de F3 (firma distinta: usa Elo en vez de nombres, documentado en el backtest).
- **Sin placeholders:** código completo en cada step.
