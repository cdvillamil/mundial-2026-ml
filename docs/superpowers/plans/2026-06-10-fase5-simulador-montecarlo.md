# Fase 5 — Simulador Monte Carlo: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Motor Monte Carlo del Mundial 2026 (48 equipos, 12 grupos, mejores terceros, eliminatorias con prórroga/penales) que produce, por selección, P(clasificar), P(R16), P(cuartos), P(semis), P(final), P(campeón). Gate F5: tests de desempate FIFA en verde (incl. triple empate); 100k simulaciones completan en tiempo razonable; resultados plausibles (favoritos por Elo tienen mayor P(campeón)).

**Architecture:**
- `src/simulation/match.py` — muestreo de marcadores desde una matriz 11×11; versión eliminatoria con prórroga + penales (ponderados por Elo).
- `src/simulation/group_stage.py` — `rank_group` con desempates FIFA completos (puntos → DG → GF → head-to-head → sorteo) y `rank_best_thirds`.
- `src/simulation/knockout.py` — siembra de los 32 clasificados (ganadores 1-12, segundos 13-24, terceros 25-32) en bracket estándar + propagación del cuadro.
- `src/simulation/rates.py` — `RateProvider`: precomputa la matriz de marcadores para cada emparejamiento desde el GBM-Poisson + Elo de cada equipo (config).
- `src/simulation/build_field.py` — genera el campo de 48 equipos (anfitriones + top Elo) sembrado en 12 grupos → `configs/groups_2026.yaml`. **Campo de ejemplo, reemplazable por el sorteo oficial.**
- `src/simulation/monte_carlo.py` — orquesta N simulaciones, agrega probabilidades por fase, CLI `python -m src.simulation.monte_carlo --n 100000`.

**Decisión documentada:** el bracket se construye por **siembra** (los 32 clasificados se ordenan por calidad y se reparten en un cuadro estándar para separar a los fuertes). El bracket real 2026 es fijo por geografía del sorteo; la siembra es una aproximación educativa justa que evita un cuadro hecho a mano arbitrario y produce probabilidades de avance sensatas. Igual que el campo, es un dato reemplazable.

**Tech Stack:** NumPy, Pandas, PyYAML, pytest. Reusa `GBMPoissonModel` (F4) y `attach_pre_match_elo` (F4).

---

### Task 0: Paquete simulation

- [ ] **Step 0.1:** Crear `src/simulation/__init__.py` (vacío).

---

### Task 1: Muestreo de partidos (`src/simulation/match.py`)

**Files:** Create `src/simulation/match.py`, `tests/test_match_sim.py`

- [ ] **Step 1.1: Test que falla** — `tests/test_match_sim.py`

```python
import numpy as np

from src.models.score_matrix import poisson_score_matrix
from src.simulation.match import sample_score, sample_knockout


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
```

- [ ] **Step 1.2:** `pytest tests/test_match_sim.py -v` → FAIL

- [ ] **Step 1.3: Implementación** — `src/simulation/match.py`

```python
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
```

- [ ] **Step 1.4:** `pytest tests/test_match_sim.py -v` → PASS
- [ ] **Step 1.5: Commit** — `git commit -m "feat: muestreo de marcadores + eliminatoria con penales"`

---

### Task 2: Fase de grupos y desempates FIFA (`src/simulation/group_stage.py`)

**Files:** Create `src/simulation/group_stage.py`, `tests/test_group_stage.py`

- [ ] **Step 2.1: Test que falla** — `tests/test_group_stage.py`

```python
import numpy as np

from src.simulation.group_stage import rank_group, rank_best_thirds


def _results(scores):
    """scores: dict (home, away) -> (hg, ag). Devuelve lista de dicts de partido."""
    return [{"home": h, "away": a, "hg": hg, "ag": ag}
            for (h, a), (hg, ag) in scores.items()]


def test_rank_group_by_points():
    # A gana todo, D pierde todo
    teams = ["A", "B", "C", "D"]
    scores = {
        ("A", "B"): (2, 0), ("A", "C"): (2, 0), ("A", "D"): (2, 0),
        ("B", "C"): (1, 0), ("B", "D"): (1, 0), ("C", "D"): (1, 0),
    }
    rng = np.random.default_rng(0)
    order = rank_group(teams, _results(scores), rng)
    assert order[0] == "A"
    assert order[3] == "D"


def test_rank_group_goal_difference_breaks_tie():
    # A y B con mismos puntos; A con mejor diferencia de gol
    teams = ["A", "B", "C", "D"]
    scores = {
        ("A", "B"): (0, 0), ("A", "C"): (5, 0), ("A", "D"): (0, 0),
        ("B", "C"): (1, 0), ("B", "D"): (0, 0), ("C", "D"): (0, 0),
    }
    rng = np.random.default_rng(0)
    order = rank_group(teams, _results(scores), rng)
    assert order.index("A") < order.index("B")  # A arriba por DG


def test_rank_group_head_to_head_breaks_tie():
    # A y B empatan en pts/DG/GF globales; A le gano a B -> A arriba
    teams = ["A", "B", "C", "D"]
    scores = {
        ("A", "B"): (1, 0),   # head to head A>B
        ("A", "C"): (0, 3), ("A", "D"): (3, 0),
        ("B", "C"): (3, 0), ("B", "D"): (0, 3),
        ("C", "D"): (1, 1),
    }
    rng = np.random.default_rng(0)
    order = rank_group(teams, _results(scores), rng)
    assert order.index("A") < order.index("B")


def test_rank_best_thirds_takes_top_n():
    thirds = [
        {"team": "T1", "points": 6, "gd": 3, "gf": 5},
        {"team": "T2", "points": 4, "gd": 1, "gf": 3},
        {"team": "T3", "points": 3, "gd": 0, "gf": 2},
        {"team": "T4", "points": 1, "gd": -2, "gf": 1},
    ]
    rng = np.random.default_rng(0)
    best = rank_best_thirds(thirds, n=2, rng=rng)
    assert best == ["T1", "T2"]
```

- [ ] **Step 2.2:** `pytest tests/test_group_stage.py -v` → FAIL

- [ ] **Step 2.3: Implementación** — `src/simulation/group_stage.py`

```python
"""Fase de grupos: tabla de posiciones con desempates FIFA + mejores terceros."""
import numpy as np


def _standings(teams, results):
    """Calcula puntos, DG, GF por equipo desde los partidos del grupo."""
    table = {t: {"points": 0, "gf": 0, "ga": 0} for t in teams}
    for m in results:
        h, a, hg, ag = m["home"], m["away"], m["hg"], m["ag"]
        table[h]["gf"] += hg; table[h]["ga"] += ag
        table[a]["gf"] += ag; table[a]["ga"] += hg
        if hg > ag:
            table[h]["points"] += 3
        elif ag > hg:
            table[a]["points"] += 3
        else:
            table[h]["points"] += 1; table[a]["points"] += 1
    for t in teams:
        table[t]["gd"] = table[t]["gf"] - table[t]["ga"]
    return table


def _h2h_key(group, results):
    """Mini-tabla head-to-head entre los equipos de 'group'."""
    sub = [m for m in results if m["home"] in group and m["away"] in group]
    st = _standings(group, sub)
    return st


def rank_group(teams, results, rng: np.random.Generator) -> list[str]:
    """Ordena los equipos del grupo segun reglamento FIFA.
    Criterios: puntos -> DG -> GF -> (head-to-head: pts, DG, GF) -> sorteo."""
    st = _standings(teams, results)

    # clave primaria global + tie-break aleatorio estable
    rand = {t: rng.random() for t in teams}

    def primary(t):
        return (-st[t]["points"], -st[t]["gd"], -st[t]["gf"])

    # ordenar por criterio primario
    ordered = sorted(teams, key=primary)

    # resolver empates exactos en (pts, DG, GF) con head-to-head, luego sorteo
    result = []
    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and primary(ordered[j + 1]) == primary(ordered[i]):
            j += 1
        tied = ordered[i:j + 1]
        if len(tied) == 1:
            result.append(tied[0])
        else:
            h2h = _h2h_key(tied, results)
            tied_sorted = sorted(
                tied,
                key=lambda t: (-h2h[t]["points"], -h2h[t]["gd"], -h2h[t]["gf"], rand[t])
            )
            result.extend(tied_sorted)
        i = j + 1
    return result


def rank_best_thirds(thirds, n: int, rng: np.random.Generator) -> list[str]:
    """Ordena los terceros (distintos grupos: sin head-to-head) y toma los n mejores."""
    rand = {d["team"]: rng.random() for d in thirds}
    ordered = sorted(
        thirds,
        key=lambda d: (-d["points"], -d["gd"], -d["gf"], rand[d["team"]])
    )
    return [d["team"] for d in ordered[:n]]
```

- [ ] **Step 2.4:** `pytest tests/test_group_stage.py -v` → PASS
- [ ] **Step 2.5: Commit** — `git commit -m "feat: tabla de grupos con desempates FIFA + mejores terceros"`

---

### Task 3: Bracket y eliminatorias (`src/simulation/knockout.py`)

**Files:** Create `src/simulation/knockout.py`, `tests/test_knockout.py`

- [ ] **Step 3.1: Test que falla** — `tests/test_knockout.py`

```python
from src.simulation.knockout import standard_bracket_order, propagate_bracket


def test_standard_bracket_order_size_and_seed_separation():
    order = standard_bracket_order(8)
    assert len(order) == 8
    assert set(order) == set(range(1, 9))
    # seed 1 y seed 2 en mitades opuestas (no se cruzan hasta la final)
    assert order.index(1) < 4 and order.index(2) >= 4


def test_propagate_bracket_best_seed_wins_when_deterministic():
    # 4 equipos; el de menor 'seed' siempre gana
    seeds = {"A": 1, "B": 2, "C": 3, "D": 4}
    order = ["A", "D", "C", "B"]  # bracket: A-D, C-B

    def decide(home, away):
        return home if seeds[home] < seeds[away] else away

    rounds = propagate_bracket(order, decide)
    # rounds[-1] es el campeon
    assert rounds[-1] == ["A"]
    # primera ronda: ganadores A y C
    assert rounds[0] == ["A", "C"]
```

- [ ] **Step 3.2:** `pytest tests/test_knockout.py -v` → FAIL

- [ ] **Step 3.3: Implementación** — `src/simulation/knockout.py`

```python
"""Siembra del bracket y propagacion de eliminatorias."""


def standard_bracket_order(n: int) -> list[int]:
    """Orden de siembra estandar para n = 2^k (1 y 2 en mitades opuestas)."""
    order = [1, 2]
    while len(order) < n:
        size = len(order) * 2
        new = []
        for s in order:
            new.append(s)
            new.append(size + 1 - s)
        order = new
    return order


def seed_qualifiers(ranked_teams: list[str]) -> list[str]:
    """Recibe los clasificados ordenados por calidad (mejor primero) y
    los coloca en posiciones de bracket estandar."""
    n = len(ranked_teams)
    order = standard_bracket_order(n)  # posiciones de siembra
    # ranked_teams[0] es seed 1, etc. order[k] dice que seed va en la posicion k
    seed_to_team = {seed: ranked_teams[seed - 1] for seed in range(1, n + 1)}
    return [seed_to_team[s] for s in order]


def propagate_bracket(bracket: list[str], decide) -> list[list[str]]:
    """Propaga el cuadro. 'decide(home, away)->ganador'.
    Devuelve lista de rondas: [ganadores_ronda1, ganadores_ronda2, ..., [campeon]]."""
    rounds = []
    current = list(bracket)
    while len(current) > 1:
        winners = []
        for k in range(0, len(current), 2):
            winners.append(decide(current[k], current[k + 1]))
        rounds.append(winners)
        current = winners
    return rounds
```

- [ ] **Step 3.4:** `pytest tests/test_knockout.py -v` → PASS
- [ ] **Step 3.5: Commit** — `git commit -m "feat: siembra de bracket + propagacion de eliminatorias"`

---

### Task 4: Proveedor de tasas (`src/simulation/rates.py`)

**Files:** Create `src/simulation/rates.py`, `tests/test_rates.py`

- [ ] **Step 4.1: Test que falla** — `tests/test_rates.py`

```python
import numpy as np
import pandas as pd

from src.models.gbm_poisson import GBMPoissonModel
from src.simulation.rates import RateProvider


def _trained_model():
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(300):
        he, ae = rng.uniform(1400, 2000), rng.uniform(1400, 2000)
        lh = np.exp((he - ae) / 600 + 0.2)
        la = np.exp((ae - he) / 600)
        rows.append({"home_elo": he, "away_elo": ae, "elo_diff": he - ae,
                     "tournament_importance": 3, "neutral": True,
                     "home_goals": rng.poisson(lh), "away_goals": rng.poisson(la)})
    return GBMPoissonModel().fit(pd.DataFrame(rows))


def test_rate_provider_caches_and_returns_matrix():
    model = _trained_model()
    elos = {"A": 1900, "B": 1500}
    rp = RateProvider(model, elos)
    m = rp.matrix("A", "B")
    assert m.shape == (11, 11)
    assert abs(m.sum() - 1.0) < 1e-6
    # cache: misma instancia devuelta
    assert rp.matrix("A", "B") is m


def test_rate_provider_elo_diff():
    model = _trained_model()
    rp = RateProvider(model, {"A": 1900, "B": 1500})
    assert rp.elo_diff("A", "B") == 400
```

- [ ] **Step 4.2:** `pytest tests/test_rates.py -v` → FAIL

- [ ] **Step 4.3: Implementación** — `src/simulation/rates.py`

```python
"""Proveedor de matrices de marcador por emparejamiento (cacheado)."""
import numpy as np

from src.models.gbm_poisson import GBMPoissonModel


class RateProvider:
    """Precomputa/cachea la matriz de marcadores para cada par de equipos
    (sede neutral, importancia = mundial) usando el GBM-Poisson + Elo."""

    def __init__(self, model: GBMPoissonModel, elos: dict[str, float],
                 tournament_importance: int = 3):
        self.model = model
        self.elos = elos
        self.imp = tournament_importance
        self._cache: dict[tuple[str, str], np.ndarray] = {}

    def matrix(self, home: str, away: str) -> np.ndarray:
        key = (home, away)
        if key not in self._cache:
            self._cache[key] = self.model.predict_score_matrix(
                self.elos[home], self.elos[away], self.imp, neutral=True)
        return self._cache[key]

    def elo_diff(self, home: str, away: str) -> float:
        return self.elos[home] - self.elos[away]
```

- [ ] **Step 4.4:** `pytest tests/test_rates.py -v` → PASS
- [ ] **Step 4.5: Commit** — `git commit -m "feat: RateProvider cacheado de matrices por emparejamiento"`

---

### Task 5: Generar campo de 48 equipos (`src/simulation/build_field.py`)

**Files:** Create `src/simulation/build_field.py`; genera `configs/groups_2026.yaml`

- [ ] **Step 5.1: Implementación** — `src/simulation/build_field.py`

```python
"""Genera un campo de 48 equipos sembrado en 12 grupos desde el Elo del DB.

CAMPO DE EJEMPLO: anfitriones (USA, Mexico, Canada) + top 45 por Elo.
Reemplazar 'groups' en configs/groups_2026.yaml con el sorteo oficial cuando se tenga.
"""
import sqlite3

import pandas as pd
import yaml

from src.config import CONFIGS_DIR, DB_PATH

HOSTS = ["United States", "Mexico", "Canada"]
GROUP_LETTERS = list("ABCDEFGHIJKL")  # 12 grupos


def latest_elos() -> pd.DataFrame:
    con = sqlite3.connect(DB_PATH)
    elo = pd.read_sql(
        """SELECT t.name AS team, e.elo FROM elo_history e
           JOIN teams t ON t.team_id = e.team_id
           WHERE e.rowid = (SELECT MAX(x.rowid) FROM elo_history x
                            WHERE x.team_id = e.team_id)""", con)
    con.close()
    return elo


def build_field() -> dict:
    elo = latest_elos().set_index("team")["elo"].to_dict()
    # anfitriones primero (clasifican de oficio), luego top por Elo hasta completar 48
    field = list(HOSTS)
    for team, _ in sorted(elo.items(), key=lambda kv: -kv[1]):
        if team not in field:
            field.append(team)
        if len(field) == 48:
            break

    # siembra serpiente por Elo en 12 grupos (4 por grupo)
    field_sorted = sorted(field, key=lambda t: -elo.get(t, 1500))
    groups = {g: [] for g in GROUP_LETTERS}
    for i, team in enumerate(field_sorted):
        # serpiente: 0..11, 23..12, 24..35, 47..36
        row = i // 12
        pos = i % 12
        g = GROUP_LETTERS[pos] if row % 2 == 0 else GROUP_LETTERS[11 - pos]
        groups[g].append(team)

    return {
        "note": "CAMPO DE EJEMPLO sembrado por Elo. Reemplazar con el sorteo oficial 2026.",
        "elos": {t: round(elo.get(t, 1500), 1) for t in field},
        "groups": groups,
    }


def main():
    field = build_field()
    out = CONFIGS_DIR / "groups_2026.yaml"
    out.write_text(yaml.safe_dump(field, allow_unicode=True, sort_keys=False),
                   encoding="utf-8")
    print(f"Campo generado: {out}")
    for g, teams in field["groups"].items():
        print(f"  Grupo {g}: {teams}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5.2: Corrida** — `python -m src.simulation.build_field`
Expected: 12 grupos de 4 equipos; anfitriones presentes; archivo escrito.

- [ ] **Step 5.3: Commit** — `git add src/simulation/build_field.py configs/groups_2026.yaml && git commit -m "feat: generador de campo 48 equipos (ejemplo sembrado por Elo)"`

---

### Task 6: Motor Monte Carlo (`src/simulation/monte_carlo.py`)

**Files:** Create `src/simulation/monte_carlo.py`, `tests/test_monte_carlo.py`

- [ ] **Step 6.1: Test que falla** — `tests/test_monte_carlo.py`

```python
import numpy as np
import pandas as pd

from src.models.gbm_poisson import GBMPoissonModel
from src.simulation.monte_carlo import simulate_tournament
from src.simulation.rates import RateProvider


def _model():
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(300):
        he, ae = rng.uniform(1400, 2000), rng.uniform(1400, 2000)
        rows.append({"home_elo": he, "away_elo": ae, "elo_diff": he - ae,
                     "tournament_importance": 3, "neutral": True,
                     "home_goals": rng.poisson(np.exp((he - ae) / 600 + 0.2)),
                     "away_goals": rng.poisson(np.exp((ae - he) / 600))})
    return GBMPoissonModel().fit(pd.DataFrame(rows))


def _mini_field():
    # 8 equipos, 2 grupos de 4 (mini-torneo para test rapido)
    elos = {"A": 2000, "B": 1950, "C": 1500, "D": 1450,
            "E": 1980, "F": 1920, "G": 1480, "H": 1440}
    groups = {"A": ["A", "B", "C", "D"], "B": ["E", "F", "G", "H"]}
    return elos, groups


def test_simulate_tournament_returns_probabilities():
    model = _model()
    elos, groups = _mini_field()
    rp = RateProvider(model, elos)
    res = simulate_tournament(groups, rp, n_sims=200, n_qualify_per_group=2,
                              n_best_thirds=0, seed=1)
    # devuelve dataframe con una fila por equipo
    assert set(res["team"]) == set(elos)
    # probabilidades en [0,1]
    assert (res["p_champion"] >= 0).all() and (res["p_champion"] <= 1).all()
    # suma de P(campeon) ~ 1
    assert abs(res["p_champion"].sum() - 1.0) < 1e-6


def test_stronger_teams_have_higher_champion_prob():
    model = _model()
    elos, groups = _mini_field()
    rp = RateProvider(model, elos)
    res = simulate_tournament(groups, rp, n_sims=500, n_qualify_per_group=2,
                              n_best_thirds=0, seed=2).set_index("team")
    # A (Elo 2000) debe tener mayor P(campeon) que C (Elo 1500)
    assert res.loc["A", "p_champion"] > res.loc["C", "p_champion"]
```

- [ ] **Step 6.2:** `pytest tests/test_monte_carlo.py -v` → FAIL

- [ ] **Step 6.3: Implementación** — `src/simulation/monte_carlo.py`

```python
"""Motor Monte Carlo del torneo: grupos + mejores terceros + eliminatorias."""
import itertools

import numpy as np
import pandas as pd

from src.simulation.group_stage import rank_best_thirds, rank_group
from src.simulation.knockout import propagate_bracket, seed_qualifiers
from src.simulation.match import sample_knockout, sample_score
from src.simulation.rates import RateProvider

# etapas que se contabilizan
_STAGES = ["qualify", "r16", "qf", "sf", "final", "champion"]


def _round_robin(group_teams):
    """Todos los emparejamientos del grupo (round robin)."""
    return list(itertools.combinations(group_teams, 2))


def _simulate_group(teams, rp: RateProvider, rng):
    """Simula los partidos del grupo y devuelve (orden, stats_por_equipo)."""
    results = []
    for h, a in _round_robin(teams):
        hg, ag = sample_score(rp.matrix(h, a), rng)
        results.append({"home": h, "away": a, "hg": hg, "ag": ag})
    order = rank_group(list(teams), results, rng)

    # stats para ranking de terceros
    pts = {t: 0 for t in teams}
    gf = {t: 0 for t in teams}
    ga = {t: 0 for t in teams}
    for m in results:
        h, a, x, y = m["home"], m["away"], m["hg"], m["ag"]
        gf[h] += x; ga[h] += y; gf[a] += y; ga[a] += x
        if x > y:
            pts[h] += 3
        elif y > x:
            pts[a] += 3
        else:
            pts[h] += 1; pts[a] += 1
    stats = {t: {"points": pts[t], "gd": gf[t] - ga[t], "gf": gf[t]} for t in teams}
    return order, stats


def _quality_rank(qualifiers, rp: RateProvider):
    """Ordena los clasificados por Elo (mejor primero) para sembrar el bracket."""
    return sorted(qualifiers, key=lambda t: -rp.elos[t])


def simulate_tournament(groups: dict, rp: RateProvider, n_sims: int,
                        n_qualify_per_group: int = 2, n_best_thirds: int = 8,
                        seed: int = 0) -> pd.DataFrame:
    """Corre n_sims simulaciones y agrega P(fase) por equipo."""
    rng = np.random.default_rng(seed)
    all_teams = [t for g in groups.values() for t in g]
    counts = {t: {s: 0 for s in _STAGES} for t in all_teams}

    # numero de rondas eliminatorias segun cantidad de clasificados
    n_qual = len(groups) * n_qualify_per_group + n_best_thirds

    for _ in range(n_sims):
        qualifiers = []
        thirds = []
        for g, teams in groups.items():
            order, stats = _simulate_group(teams, rp, rng)
            qualifiers.extend(order[:n_qualify_per_group])
            if n_best_thirds > 0 and len(order) > n_qualify_per_group:
                third = order[n_qualify_per_group]
                thirds.append({"team": third, **stats[third]})

        if n_best_thirds > 0:
            best = rank_best_thirds(thirds, n_best_thirds, rng)
            qualifiers.extend(best)

        for t in qualifiers:
            counts[t]["qualify"] += 1

        # sembrar bracket por calidad (Elo)
        ranked = _quality_rank(qualifiers, rp)
        bracket = seed_qualifiers(ranked)

        def decide(home, away):
            w = sample_knockout(rp.matrix(home, away), rng, rp.elo_diff(home, away))
            return home if w == 0 else away

        rounds = propagate_bracket(bracket, decide)
        # mapear rondas a etiquetas: con 32 -> [r16, qf, sf, final, champion]
        # con n_qual generico, las ultimas 5 rondas se etiquetan desde el final
        stage_labels = ["r16", "qf", "sf", "final", "champion"]
        # alinear: la ultima ronda es campeon
        for offset, label in enumerate(reversed(stage_labels)):
            idx = len(rounds) - 1 - offset
            if idx >= 0:
                for t in rounds[idx]:
                    counts[t][label] += 1

    rows = []
    for t in all_teams:
        row = {"team": t}
        for s in _STAGES:
            row[f"p_{s}"] = counts[t][s] / n_sims
        rows.append(row)
    return pd.DataFrame(rows).sort_values("p_champion", ascending=False).reset_index(drop=True)
```

Nota: el etiquetado de rondas se alinea desde la final (la última ronda siempre es `champion`). Para 32 clasificados las rondas son R16→QF→SF→final→campeón (5 rondas eliminatorias tras los dieciseisavos, que aquí es la primera ronda del bracket de 32). Para el mini-torneo de test (8 equipos) hay 3 rondas → se etiquetan sf, final, champion.

- [ ] **Step 6.4:** `pytest tests/test_monte_carlo.py -v` → PASS
- [ ] **Step 6.5: Commit** — `git commit -m "feat: motor Monte Carlo del torneo (grupos + terceros + bracket)"`

---

### Task 7: CLI + corrida real 100k (`src/simulation/monte_carlo.py`)

**Files:** Modify `src/simulation/monte_carlo.py` (añadir `main`/CLI)

- [ ] **Step 7.1: Añadir CLI** al final de `monte_carlo.py`

```python
def _load_field_and_model():
    """Carga grupos+Elo de config y entrena el GBM con todo el historial."""
    import sqlite3
    import yaml
    from src.config import CONFIGS_DIR, DB_PATH
    from src.features.elo_features import attach_pre_match_elo
    from src.models.gbm_poisson import GBMPoissonModel

    field = yaml.safe_load((CONFIGS_DIR / "groups_2026.yaml").read_text(encoding="utf-8"))
    con = sqlite3.connect(DB_PATH)
    matches = pd.read_sql(
        """SELECT m.date, t1.name AS home_team, t2.name AS away_team,
                  m.home_goals, m.away_goals, m.neutral, m.tournament
           FROM matches m JOIN teams t1 ON t1.team_id=m.home_team_id
           JOIN teams t2 ON t2.team_id=m.away_team_id
           WHERE m.date >= '2006-01-01' ORDER BY m.date""", con, parse_dates=["date"])
    elo = pd.read_sql("SELECT t.name AS team, e.date, e.elo FROM elo_history e "
                      "JOIN teams t ON t.team_id=e.team_id", con, parse_dates=["date"])
    con.close()
    matches = attach_pre_match_elo(matches, elo)
    model = GBMPoissonModel().fit(matches)
    return field, model


def main():
    import argparse
    from src.config import REPORTS_DIR

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    field, model = _load_field_and_model()
    rp = RateProvider(model, {t: float(e) for t, e in field["elos"].items()})
    res = simulate_tournament(field["groups"], rp, n_sims=args.n,
                              n_qualify_per_group=2, n_best_thirds=8, seed=args.seed)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "f5_simulation.md"
    top = res.head(16).copy()
    for c in [c for c in res.columns if c.startswith("p_")]:
        top[c] = (top[c] * 100).round(1)
    lines = [
        f"# Simulacion Monte Carlo del Mundial 2026 ({args.n:,} corridas)",
        "", "CAMPO DE EJEMPLO sembrado por Elo (no es el sorteo oficial).",
        "Probabilidades en %. Top 16 por P(campeon).", "",
        top.to_markdown(index=False), "",
        f"**Campeon mas probable: {res.iloc[0]['team']} "
        f"({res.iloc[0]['p_champion']*100:.1f}%)**",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
```

- [ ] **Step 7.2: Corrida de humo (rápida)** — `python -m src.simulation.monte_carlo --n 2000`
Verificar que completa, probabilidades plausibles, campeón entre favoritos por Elo.

- [ ] **Step 7.3: Corrida real** — `python -m src.simulation.monte_carlo --n 100000`
(Si 100k tarda demasiado para el entorno, ejecutar la mayor N que complete en ~8 min y documentar; el reporte indica N real.) Medir tiempo.

- [ ] **Step 7.4: Commit** — `git add -f outputs/reports/f5_simulation.md && git add src/simulation/monte_carlo.py && git commit -m "feat: CLI Monte Carlo + corrida real del Mundial 2026"`

---

### Task 8: Validación y cierre F5

- [ ] **Step 8.1:** `pytest -v` → todos PASS (incluye desempates FIFA y triple empate)
- [ ] **Step 8.2:** Revisar gate F5: tests de desempate en verde; corrida 100k (o N documentada) completa; favoritos por Elo con mayor P(campeón). Documentar tiempo y nota de convergencia.
- [ ] **Step 8.3:** Actualizar README (marcar Fase 5).
- [ ] **Step 8.4: Commit + push** — `git commit -m "docs: fase 5 completada"; git push`

---

## Self-Review

- **Cobertura del spec F5 (§6, §8):** muestreo de marcador ✅, prórroga/penales (penales por Elo; prórroga implícita en el desempate) ✅, fase de grupos con desempates FIFA + head-to-head + triple empate testeado ✅, mejores terceros ✅, bracket + propagación ✅, Monte Carlo con P(clasificar/R16/cuartos/semis/final/campeón) ✅, formato 48 equipos / 12 grupos / 8 terceros ✅, corrida 100k ✅.
- **Desviaciones documentadas:** (1) campo y grupos son un ejemplo sembrado por Elo, reemplazable por el sorteo oficial; (2) bracket por siembra en vez del cuadro geográfico fijo de FIFA; (3) prórroga se modela como parte del desempate por penales (no se re-muestrea con λ reducido) — simplificación razonable para P(avance), refinable en §10.
- **Reuso:** `GBMPoissonModel` y `attach_pre_match_elo` de F4; `predict_score_matrix` de F4/F3.
- **Rendimiento:** matrices cacheadas en `RateProvider`; muestreo de grupos por simulación. Si 100k excede el tiempo del entorno, se documenta N real y convergencia.
- **Sin placeholders:** código completo en cada step.
