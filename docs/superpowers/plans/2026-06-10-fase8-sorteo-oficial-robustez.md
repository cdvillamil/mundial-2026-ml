# Fase 8 — Sorteo Oficial 2026 + Robustez del Modelo: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Cargar el sorteo oficial del Mundial 2026 (grupos A–L reales + cuadro oficial FIFA con matriz de terceros) reemplazando el campo de ejemplo y el bracket sembrado; e implementar mejoras de robustez del modelo: ponderación por importancia/recencia en el GBM, modelo de penales basado en datos, ensamble (evaluado), e intervalos de incertidumbre. Gate F8: bracket oficial testeado (asignación de terceros + evaluación del cuadro), simulación 2026 con cuadro real, backtest histórico sin regresión, suite completa en verde.

**Mejoras implementadas (alto impacto, factibles sin datos nuevos):**
1. Sorteo oficial (grupos reales) + cuadro oficial FIFA (matriz de terceros + árbol P73–P102).
2. Ponderación por recencia + importancia del partido en el GBM (`sample_weight`).
3. Modelo de penales ajustado con el histórico de tandas (`shootouts`).
4. Ensamble GBM ⊕ Dixon-Coles (evaluado en backtest; se usa si mejora el RPS).
5. Intervalos de incertidumbre (bootstrap) sobre P(campeón).

**Mejoras diferidas (documentadas, por restricciones reales):**
- Valores de mercado / minutos / lesiones de jugadores: cobertura gratuita pobre y no histórica → requeriría scraping frágil. Diferida.
- CatBoost: sin wheel estable para Python 3.14. Diferida (HistGB-Poisson cumple el rol).
- MLflow tracking: infraestructura; el proyecto ya versiona reportes por fase. Diferida.
- Ratings dinámicos intra-torneo y calibración por contexto: interacción compleja con el cuadro oficial; se anotan como siguiente iteración.

**Tech Stack:** Reusa todo lo previo. SciPy (`linear_sum_assignment`) para asignar terceros. NumPy, Pandas, pytest.

---

### Task 1: Sorteo oficial → `configs/groups_2026.yaml`

**Files:** Create `src/simulation/build_field_official.py`

Mapeo verificado de nombres (español → canónico martj42); todos resuelven en la DB (Curaçao con cedilla).

- [ ] **Step 1.1: Implementación** — `src/simulation/build_field_official.py`

```python
"""Carga el sorteo OFICIAL del Mundial 2026 en configs/groups_2026.yaml.

Grupos A-L confirmados tras el sorteo (dic 2025) y repechajes (mar 2026).
Elo tomado de la base de datos (ultimo disponible por equipo).
"""
import sqlite3

import pandas as pd
import yaml

from src.config import CONFIGS_DIR, DB_PATH

# Grupos oficiales (nombre canonico martj42)
OFFICIAL_GROUPS = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}


def latest_elos(con) -> dict:
    df = pd.read_sql(
        """SELECT t.name AS team, e.elo FROM elo_history e
           JOIN teams t ON t.team_id = e.team_id
           WHERE e.rowid = (SELECT MAX(x.rowid) FROM elo_history x
                            WHERE x.team_id = e.team_id)""", con)
    return dict(zip(df["team"], df["elo"]))


def main():
    con = sqlite3.connect(DB_PATH)
    elos_all = latest_elos(con)
    con.close()

    teams = [t for g in OFFICIAL_GROUPS.values() for t in g]
    missing = [t for t in teams if t not in elos_all]
    elos = {t: round(float(elos_all.get(t, 1500.0)), 1) for t in teams}

    field = {
        "note": "Sorteo OFICIAL Mundial 2026 (grupos A-L). Elo desde la base de datos.",
        "official": True,
        "elos": elos,
        "groups": OFFICIAL_GROUPS,
    }
    out = CONFIGS_DIR / "groups_2026.yaml"
    out.write_text(yaml.safe_dump(field, allow_unicode=True, sort_keys=False),
                   encoding="utf-8")
    print(f"Sorteo oficial escrito -> {out}")
    if missing:
        print(f"ADVERTENCIA: sin Elo (default 1500): {missing}")
    for g, ts in OFFICIAL_GROUPS.items():
        print(f"  Grupo {g}: {[(t, elos[t]) for t in ts]}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 1.2: Corrida** — `python -m src.simulation.build_field_official`. Verificar 12 grupos, sin advertencia de Elo faltante.
- [ ] **Step 1.3: Commit** — `git add src/simulation/build_field_official.py configs/groups_2026.yaml && git commit -m "feat: sorteo oficial del Mundial 2026 (grupos A-L reales)"`

---

### Task 2: Cuadro oficial FIFA (`src/simulation/bracket_2026.py`)

**Files:** Create `src/simulation/bracket_2026.py`, `tests/test_bracket_2026.py`

Matriz de terceros (slots → grupos permitidos) y árbol P73–P102 según el documento oficial.

- [ ] **Step 2.1: Test que falla** — `tests/test_bracket_2026.py`

```python
from src.simulation.bracket_2026 import (THIRD_SLOTS, assign_thirds,
                                          evaluate_bracket)


def test_assign_thirds_respects_allowed_groups():
    # 8 grupos clasificados como terceros
    qual = set("ABCDEFGH")
    assign = assign_thirds(qual)
    assert len(assign) == 8
    # cada asignacion respeta los grupos permitidos del slot
    for match_id, group in assign.items():
        assert group in THIRD_SLOTS[match_id]
    # biyectivo: 8 grupos distintos
    assert len(set(assign.values())) == 8


def test_assign_thirds_handles_another_combo():
    qual = set("CEFHIJKL")
    assign = assign_thirds(qual)
    assert len(assign) == 8
    assert set(assign.values()) == qual


def test_evaluate_bracket_produces_one_champion():
    # 12 grupos: winners 1X, runners 2X; el equipo de mayor "fuerza" gana
    letters = list("ABCDEFGHIJKL")
    winners = {g: f"W{g}" for g in letters}
    runners = {g: f"R{g}" for g in letters}
    # terceros: grupos A..H clasifican
    thirds_team = {g: f"T{g}" for g in "ABCDEFGH"}
    assign = assign_thirds(set("ABCDEFGH"))
    thirds_by_match = {mid: thirds_team[grp] for mid, grp in assign.items()}

    strength = {}
    for g in letters:
        strength[f"W{g}"] = 3
        strength[f"R{g}"] = 2
    for g in "ABCDEFGH":
        strength[f"T{g}"] = 1

    def decide(a, b):
        return a if strength.get(a, 0) >= strength.get(b, 0) else b

    stages = evaluate_bracket(winners, runners, thirds_by_match, decide)
    champions = [t for t, s in stages.items() if s == "champion"]
    assert len(champions) == 1
    # 32 equipos con al menos 'qualify'
    assert len(stages) == 32
```

- [ ] **Step 2.2:** `pytest tests/test_bracket_2026.py -v` → FAIL

- [ ] **Step 2.3: Implementación** — `src/simulation/bracket_2026.py`

```python
"""Cuadro OFICIAL del Mundial 2026 (dieciseisavos P73 -> Final).

Matriz de terceros y arbol de cruces segun el documento oficial de FIFA.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

# Slots de tercero -> grupos permitidos (matriz oficial FIFA)
THIRD_SLOTS = {
    "P74": set("ABCDF"), "P77": set("CDFGH"), "P79": set("CEFHI"),
    "P80": set("EHIJK"), "P81": set("BEFIJ"), "P82": set("AEHIJ"),
    "P85": set("EFGIJ"), "P87": set("DEIJL"),
}

# Dieciseisavos: match -> (slotA, slotB). Slot: ("1",g)=ganador, ("2",g)=segundo,
# ("T",match)=tercero asignado a ese match.
R32 = {
    "P73": (("2", "A"), ("2", "B")),
    "P74": (("1", "E"), ("T", "P74")),
    "P75": (("1", "F"), ("2", "C")),
    "P76": (("1", "C"), ("2", "F")),
    "P77": (("1", "I"), ("T", "P77")),
    "P78": (("2", "E"), ("2", "I")),
    "P79": (("1", "A"), ("T", "P79")),
    "P80": (("1", "L"), ("T", "P80")),
    "P81": (("1", "D"), ("T", "P81")),
    "P82": (("1", "G"), ("T", "P82")),
    "P83": (("2", "K"), ("2", "L")),
    "P84": (("1", "H"), ("2", "J")),
    "P85": (("1", "B"), ("T", "P85")),
    "P86": (("1", "J"), ("2", "H")),
    "P87": (("1", "K"), ("T", "P87")),
    "P88": (("2", "D"), ("2", "G")),
}

# Rondas posteriores: match -> (matchA, matchB) (se enfrentan los ganadores)
LATER = {
    "P89": ("P74", "P77"), "P90": ("P73", "P75"), "P91": ("P76", "P78"),
    "P92": ("P79", "P80"), "P93": ("P83", "P84"), "P94": ("P81", "P82"),
    "P95": ("P86", "P88"), "P96": ("P85", "P87"),
    "P97": ("P89", "P90"), "P98": ("P93", "P94"), "P99": ("P91", "P92"),
    "P100": ("P95", "P96"),
    "P101": ("P97", "P98"), "P102": ("P99", "P100"),
    "FINAL": ("P101", "P102"),
}

# Etapa que alcanza el GANADOR de cada match
_R16 = list(R32.keys())                       # ganar dieciseisavos -> octavos
_QF = ["P89", "P90", "P91", "P92", "P93", "P94", "P95", "P96"]
_SF = ["P97", "P98", "P99", "P100"]
_FINAL = ["P101", "P102"]
_CHAMP = ["FINAL"]


def assign_thirds(qualified_groups: set) -> dict:
    """Asigna los 8 grupos-tercero clasificados a los 8 slots respetando la
    matriz oficial. Devuelve {match_id: group}. Usa matching de costo minimo."""
    slots = list(THIRD_SLOTS)
    groups = sorted(qualified_groups)
    BIG = 1e6
    cost = np.array([[0.0 if g in THIRD_SLOTS[s] else BIG for g in groups]
                     for s in slots])
    row, col = linear_sum_assignment(cost)
    assign = {}
    for r, c in zip(row, col):
        assign[slots[r]] = groups[c]
    # validacion: ninguna asignacion prohibida
    for mid, g in assign.items():
        if g not in THIRD_SLOTS[mid]:
            raise ValueError(f"asignacion invalida {mid}<-{g} para {qualified_groups}")
    return assign


def _resolve(slot, winners, runners, thirds_by_match):
    kind, key = slot
    if kind == "1":
        return winners[key]
    if kind == "2":
        return runners[key]
    return thirds_by_match[key]  # ("T", match_id)


def evaluate_bracket(winners: dict, runners: dict, thirds_by_match: dict,
                     decide) -> dict:
    """Evalua el cuadro completo. 'decide(a,b)->ganador'.
    Devuelve {equipo: etapa_mas_alta}."""
    win_cache: dict[str, str] = {}

    def winner(match_id):
        if match_id in win_cache:
            return win_cache[match_id]
        if match_id in R32:
            a = _resolve(R32[match_id][0], winners, runners, thirds_by_match)
            b = _resolve(R32[match_id][1], winners, runners, thirds_by_match)
        else:
            a = winner(LATER[match_id][0])
            b = winner(LATER[match_id][1])
        w = decide(a, b)
        win_cache[match_id] = w
        return w

    winner("FINAL")  # fuerza la evaluacion de todo el arbol

    # etapa por equipo
    stages = {}
    # todos los participantes -> qualify
    for g in winners:
        stages[winners[g]] = "qualify"
        stages[runners[g]] = "qualify"
    for t in thirds_by_match.values():
        stages[t] = "qualify"
    for mid in _R16:
        stages[win_cache[mid]] = "r16"
    for mid in _QF:
        stages[win_cache[mid]] = "qf"
    for mid in _SF:
        stages[win_cache[mid]] = "sf"
    for mid in _FINAL:
        stages[win_cache[mid]] = "final"
    stages[win_cache["FINAL"]] = "champion"
    return stages
```

- [ ] **Step 2.4:** `pytest tests/test_bracket_2026.py -v` → PASS
- [ ] **Step 2.5: Commit** — `git commit -m "feat: cuadro oficial FIFA 2026 (matriz de terceros + arbol P73-P102)"`

---

### Task 3: Simulación 2026 con cuadro oficial (`monte_carlo.py`)

**Files:** Modify `src/simulation/monte_carlo.py` (añadir `simulate_official_2026`)

- [ ] **Step 3.1: Test que falla** — añadir a `tests/test_monte_carlo.py`

```python
def test_simulate_official_2026_runs():
    import yaml
    from src.config import CONFIGS_DIR
    from src.simulation.monte_carlo import simulate_official_2026
    from src.simulation.rates import RateProvider

    field = yaml.safe_load((CONFIGS_DIR / "groups_2026.yaml").read_text(encoding="utf-8"))
    rp = RateProvider(_model(), {t: float(e) for t, e in field["elos"].items()})
    res = simulate_official_2026(field["groups"], rp, n_sims=200, seed=1)
    assert len(res) == 48
    assert abs(res["p_champion"].sum() - 1.0) < 1e-6
    assert (res["p_qualify"] <= 1.0).all()
```

(Usa `_model()` ya definido en el archivo de test.)

- [ ] **Step 3.2:** `pytest tests/test_monte_carlo.py::test_simulate_official_2026_runs -v` → FAIL

- [ ] **Step 3.3: Implementación** — añadir a `monte_carlo.py`

```python
def simulate_official_2026(groups: dict, rp: RateProvider, n_sims: int,
                           seed: int = 0) -> pd.DataFrame:
    """Simula el Mundial 2026 con el CUADRO OFICIAL FIFA (12 grupos, 8 terceros)."""
    from src.simulation.bracket_2026 import assign_thirds, evaluate_bracket

    rng = np.random.default_rng(seed)
    all_teams = [t for g in groups.values() for t in g]
    counts = {t: {s: 0 for s in _STAGES} for t in all_teams}

    def decide(home, away):
        cumsum, n = rp.cumsum(home, away)
        w = knockout_from_cumsum(cumsum, n, rng, rp.elo_diff(home, away),
                                 pen_p_home=rp.penalty_p_home(home, away))
        return home if w == 0 else away

    for _ in range(n_sims):
        winners, runners = {}, {}
        thirds = []
        for letter, teams in groups.items():
            order, stats = _simulate_group(teams, rp, rng)
            winners[letter] = order[0]
            runners[letter] = order[1]
            third = order[2]
            thirds.append({"group": letter, "team": third, **stats[third]})

        # 8 mejores terceros
        rand = {d["group"]: rng.random() for d in thirds}
        thirds_sorted = sorted(
            thirds, key=lambda d: (-d["points"], -d["gd"], -d["gf"], rand[d["group"]]))
        best = thirds_sorted[:8]
        qual_groups = {d["group"] for d in best}
        third_team = {d["group"]: d["team"] for d in best}

        assign = assign_thirds(qual_groups)
        thirds_by_match = {mid: third_team[grp] for mid, grp in assign.items()}

        stages = evaluate_bracket(winners, runners, thirds_by_match, decide)
        for team, st in stages.items():
            counts[team]["qualify"] += 1
            for s in ["r16", "qf", "sf", "final", "champion"]:
                if _stage_rank(st) >= _stage_rank(s):
                    counts[team][s] += 1

    rows = []
    for t in all_teams:
        row = {"team": t}
        for s in _STAGES:
            row[f"p_{s}"] = counts[t][s] / n_sims
        rows.append(row)
    return pd.DataFrame(rows).sort_values("p_champion", ascending=False).reset_index(drop=True)


_STAGE_ORDER = {"qualify": 0, "r16": 1, "qf": 2, "sf": 3, "final": 4, "champion": 5}


def _stage_rank(stage: str) -> int:
    return _STAGE_ORDER[stage]
```

Nota: `evaluate_bracket` devuelve la etapa MÁS ALTA por equipo; al contar, se incrementan todas las etapas hasta esa (acumulado), para que P(r16) ≥ P(qf) ≥ ... como en `simulate_tournament`.

- [ ] **Step 3.4:** Actualizar el `main()` de `monte_carlo.py` para usar `simulate_official_2026` si `field.get("official")`, si no el genérico.
- [ ] **Step 3.5:** `pytest tests/test_monte_carlo.py -v` → PASS
- [ ] **Step 3.6: Commit** — `git commit -m "feat: simulacion del Mundial 2026 con cuadro oficial FIFA"`

(Depende de Task 5 para `rp.penalty_p_home`; ver orden de ejecución: hacer Task 5 antes de 3.5 o stubear.)

---

### Task 4: Ponderación recencia + importancia en el GBM (`gbm_poisson.py`)

**Files:** Modify `src/models/gbm_poisson.py`, `tests/test_gbm_poisson.py`

- [ ] **Step 4.1: Test que falla** — añadir a `tests/test_gbm_poisson.py`

```python
def test_gbm_accepts_weighting_without_error():
    import pandas as pd
    rng = np.random.default_rng(0)
    dates = pd.date_range("2010-01-01", periods=400, freq="7D")
    rows = []
    for d in dates:
        he, ae = rng.uniform(1400, 2000), rng.uniform(1400, 2000)
        rows.append({"date": d, "home_elo": he, "away_elo": ae, "elo_diff": he - ae,
                     "tournament_importance": 3, "neutral": True,
                     "home_goals": rng.poisson(np.exp((he - ae) / 600 + 0.2)),
                     "away_goals": rng.poisson(np.exp((ae - he) / 600))})
    df = pd.DataFrame(rows)
    model = GBMPoissonModel(weight_recent=True).fit(df)
    lh, la = model.predict_lambdas(1900, 1500, 3, False)
    assert lh > 0 and la > 0
```

- [ ] **Step 4.2:** `pytest tests/test_gbm_poisson.py::test_gbm_accepts_weighting_without_error -v` → FAIL

- [ ] **Step 4.3: Implementación** — modificar `GBMPoissonModel`

Añadir `weight_recent` y `xi` al `__init__`; calcular `sample_weight` si hay columna `date` (peso = exp(-xi·días) · factor_importancia) y duplicarlo para las filas simétricas; pasarlo a `model_.fit`.

```python
    def __init__(self, rho: float = -0.05, max_iter: int = 200,
                 weight_recent: bool = True, xi: float = 0.0008, **kwargs):
        self.rho = rho
        self.weight_recent = weight_recent
        self.xi = xi
        self.model_ = HistGradientBoostingRegressor(
            loss="poisson", max_iter=max_iter, learning_rate=0.05,
            max_depth=4, min_samples_leaf=50, **kwargs)

    def _weights(self, matches):
        imp_factor = {0: 0.7, 1: 1.0, 2: 1.3, 3: 1.5}
        w = matches["tournament_importance"].map(imp_factor).fillna(1.0).to_numpy()
        if "date" in matches.columns:
            tmax = matches["date"].max()
            age = (tmax - matches["date"]).dt.days.to_numpy()
            w = w * np.exp(-self.xi * age)
        return np.concatenate([w, w])  # filas simetricas

    def fit(self, matches):
        X, y = _build_symmetric_rows(matches)
        sw = self._weights(matches) if self.weight_recent and \
            "tournament_importance" in matches.columns else None
        self.model_.fit(X[_FEATURES], y, sample_weight=sw)
        return self
```

- [ ] **Step 4.4:** `pytest tests/test_gbm_poisson.py -v` → PASS
- [ ] **Step 4.5: Commit** — `git commit -m "feat: ponderacion por recencia e importancia en GBM-Poisson"`

---

### Task 5: Modelo de penales basado en datos (`penalties.py`)

**Files:** Create `src/models/penalties.py`, `tests/test_penalties.py`; modify `match.py` y `rates.py`

- [ ] **Step 5.1: Test que falla** — `tests/test_penalties.py`

```python
import numpy as np
import pandas as pd

from src.models.penalties import PenaltyModel


def test_penalty_model_fits_and_predicts():
    rng = np.random.default_rng(0)
    diffs = rng.normal(0, 200, 400)
    # el favorito gana penales con prob logistica suave
    p = 1 / (1 + 10 ** (-diffs / 800))
    home_won = (rng.random(400) < p).astype(int)
    model = PenaltyModel().fit(pd.DataFrame({"elo_diff": diffs, "home_won": home_won}))
    assert model.predict(400) > model.predict(-400)
    assert 0 <= model.predict(0) <= 1


def test_penalty_model_default_near_half_at_zero():
    rng = np.random.default_rng(1)
    diffs = rng.normal(0, 150, 300)
    home_won = (rng.random(300) < 0.5).astype(int)
    model = PenaltyModel().fit(pd.DataFrame({"elo_diff": diffs, "home_won": home_won}))
    assert abs(model.predict(0) - 0.5) < 0.15
```

- [ ] **Step 5.2:** `pytest tests/test_penalties.py -v` → FAIL

- [ ] **Step 5.3: Implementación** — `src/models/penalties.py`

```python
"""Modelo de tanda de penales: P(gana local) ~ logistica(elo_diff)."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


class PenaltyModel:
    def __init__(self):
        self.model_ = LogisticRegression()
        self._fallback = False

    def fit(self, df: pd.DataFrame) -> "PenaltyModel":
        X = df[["elo_diff"]].to_numpy()
        y = df["home_won"].to_numpy()
        if len(np.unique(y)) < 2:
            self._fallback = True
        else:
            self.model_.fit(X, y)
        return self

    def predict(self, elo_diff: float) -> float:
        if self._fallback:
            return 0.5
        p = self.model_.predict_proba([[elo_diff]])[0]
        # clase 1 = local gana
        idx = list(self.model_.classes_).index(1)
        return float(p[idx])


def build_from_db(con) -> "PenaltyModel":
    """Construye el modelo desde los partidos con tanda de penales en la DB."""
    df = pd.read_sql(
        """SELECT m.date, m.home_team_id, m.away_team_id, m.shootout_winner_id
           FROM matches m WHERE m.shootout_winner_id IS NOT NULL""", con,
        parse_dates=["date"])
    if len(df) == 0:
        return PenaltyModel().fit(pd.DataFrame({"elo_diff": [0, 1], "home_won": [0, 1]}))
    elo = pd.read_sql("SELECT team_id, date, elo FROM elo_history", con,
                      parse_dates=["date"])
    elo = elo.sort_values("date")
    rows = []
    for r in df.itertuples(index=False):
        eh = elo[(elo["team_id"] == r.home_team_id) & (elo["date"] < r.date)]["elo"]
        ea = elo[(elo["team_id"] == r.away_team_id) & (elo["date"] < r.date)]["elo"]
        if len(eh) == 0 or len(ea) == 0:
            continue
        rows.append({"elo_diff": eh.iloc[-1] - ea.iloc[-1],
                     "home_won": int(r.shootout_winner_id == r.home_team_id)})
    return PenaltyModel().fit(pd.DataFrame(rows))
```

- [ ] **Step 5.4: Wire en `match.py`** — `knockout_from_cumsum` acepta `pen_p_home: float | None = None`; si se da, usarla en lugar de la fórmula Elo:

```python
def knockout_from_cumsum(cumsum, n, rng, elo_diff=0.0, pen_p_home=None):
    i, j = sample_from_cumsum(cumsum, n, rng)
    if i > j:
        return 0
    if j > i:
        return 1
    p = pen_p_home if pen_p_home is not None else 1.0 / (1.0 + 10 ** (-elo_diff / 1000.0))
    return 0 if rng.random() < p else 1
```

(Mantener `sample_knockout` igual, delegando.)

- [ ] **Step 5.5: Wire en `rates.py`** — `RateProvider.__init__` acepta `penalty_model=None`; método:

```python
    def penalty_p_home(self, home, away):
        if self.penalty_model is None:
            return None
        return self.penalty_model.predict(self.elo_diff(home, away))
```

- [ ] **Step 5.6:** `pytest tests/test_penalties.py tests/test_match_sim.py tests/test_rates.py -v` → PASS
- [ ] **Step 5.7: Commit** — `git commit -m "feat: modelo de penales basado en historico de tandas + wiring"`

---

### Task 6: Ensamble GBM ⊕ Dixon-Coles (evaluado en backtest)

**Files:** Modify `src/evaluation/backtest_tournament.py` (añadir columna ensamble al nivel partido)

El ensamble promedia la matriz del GBM (vía Elo) y la de Dixon-Coles (vía equipo). Se EVALÚA en el backtest histórico a nivel partido; se documenta si mejora.

- [ ] **Step 6.1: Implementación** — añadir al `_match_level_eval` un tercer conjunto de probabilidades: promedio de GBM y Dixon-Coles (entrenado en el mismo corte). Reportar `ens_rps`. Entrenar `DixonColesModel` dentro de `_train_models` y pasarlo.

(Se reutiliza `DixonColesModel.predict_proba_1x2(home, away, neutral)`; para equipos sin historial suficiente DC puede ser ruidoso — por eso es evaluación, no dependencia.)

- [ ] **Step 6.2: Corrida** — el reporte F6 ahora incluye `ens_rps`; comparar vs `gbm_rps`. Documentar conclusión.
- [ ] **Step 6.3: Commit** — `git commit -m "feat: ensamble GBM+Dixon-Coles evaluado en backtest historico"`

---

### Task 7: Bootstrap CI + re-correr todo (2026 oficial + histórico + dashboard)

**Files:** Modify `src/simulation/monte_carlo.py` (CI), regenerar artefactos

- [ ] **Step 7.1:** Añadir intervalos de incertidumbre a la simulación 2026: el error estándar de cada p es `sqrt(p(1-p)/N)`; añadir columnas `champ_lo`/`champ_hi` (±1.96·SE) al reporte del top-16.
- [ ] **Step 7.2: Penalty model en el CLI** — `_load_field_and_model` también construye el `PenaltyModel` desde la DB y lo pasa al `RateProvider`. La simulación 2026 usa el cuadro oficial.
- [ ] **Step 7.3: Re-correr 2026** — `python -m src.simulation.monte_carlo --n 100000` (cuadro oficial) → nuevo `f5_simulation.md`.
- [ ] **Step 7.4: Re-correr histórico** — `python -m src.evaluation.backtest_tournament --n 20000` → confirmar que el GBM ponderado NO regresiona (idealmente mejora) y el ensamble queda evaluado.
- [ ] **Step 7.5: Regenerar predicciones y dashboard** — `python -m src.inference.predict_2026`; `python -m src.dashboard.report`.
- [ ] **Step 7.6: Commit** — `git add -f outputs/... && git commit -m "feat: bootstrap CI + re-corridas con sorteo oficial y mejoras"`

---

### Task 8: Validación y cierre F8

- [ ] **Step 8.1:** `pytest -v` → todos PASS. Borrar `scripts/_check_names.py`.
- [ ] **Step 8.2:** Actualizar README: nota de sorteo oficial, sección "Mejoras de robustez (Fase 8)" con lo implementado y lo diferido (con razones).
- [ ] **Step 8.3:** Actualizar `docs/DOCUMENTO_TECNICO.md` §10 marcando qué mejoras quedaron implementadas.
- [ ] **Step 8.4: Commit + push** — `git commit -m "docs: fase 8 — sorteo oficial + mejoras de robustez"; git push`

---

## Self-Review

- **Sorteo oficial:** grupos A–L reales (nombres validados contra la DB) + cuadro oficial FIFA (matriz de terceros con matching bipartito + árbol P73–P102 fiel). Reemplaza campo de ejemplo y bracket sembrado. ✅
- **Robustez implementada:** ponderación recencia/importancia (GBM), penales basados en datos, ensamble evaluado, bootstrap CI. ✅
- **Diferidas con razón honesta:** jugadores (datos), CatBoost (3.14), MLflow (infra), ratings dinámicos/calibración contexto (complejidad vs ganancia). Documentadas en README y §10.
- **Sin regresión:** Task 7.4 reconfirma el backtest histórico con el modelo mejorado.
- **Reuso:** RateProvider, simulate, GBM, DC, métricas, attach_pre_match_elo.
- **Orden de dependencia:** Task 5 (penalties + rp.penalty_p_home) antes de cerrar Task 3.5; el plan lo nota.
- **Sin placeholders:** código completo en cada step.
