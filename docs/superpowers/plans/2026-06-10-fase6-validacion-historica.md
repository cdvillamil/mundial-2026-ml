# Fase 6 — Validación Histórica: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Backtesting del sistema completo contra los Mundiales 2010, 2014, 2018 y 2022: entrenar solo con datos anteriores a cada torneo, predecir partido a partido y simular el torneo entero, comparando contra el campeón y clasificados reales. Gate F6: (a) el GBM-Poisson supera el RPS del baseline Elo en ≥3 de 4 mundiales a nivel partido; (b) el campeón real queda en el top-5 de P(campeón) en ≥3 de 4 mundiales; (c) informe reproducible con un comando.

**Architecture:**
- `src/evaluation/historical.py` — utilidades sobre la base de datos:
  - `wc_matches(con, year)` — partidos del Mundial de ese año con resultado.
  - `extract_groups(matches_year)` — reconstruye los 8 grupos de 4 desde la fase de grupos (primeros 48 partidos): grafo de "jugaron entre sí" → 8 componentes conexas.
  - `champion_of(con, year)` — ganador de la final (último partido; penales si empate).
  - `elos_before(con, date, teams)` — Elo point-in-time de cada participante justo antes del torneo.
- `src/evaluation/backtest_tournament.py` — por cada mundial: entrena GBM-Poisson (corte temporal estricto) + baseline Elo, evalúa a nivel partido (RPS/log-loss/accuracy vs baseline) y simula el torneo (formato 32 equipos: 8 grupos, 2 clasifican, sin terceros) → ranking de P(campeón) y posición del campeón real. Escribe `outputs/reports/f6_historical_validation.md`.

**Decisiones documentadas:**
- El simulador usa el formato histórico (32 equipos, 8 grupos, top-2, sin mejores terceros) — el motor de Fase 5 ya es parametrizable (`n_best_thirds=0`).
- Bracket por siembra (consistente con Fase 5) en vez del cuadro fijo real. Para la pregunta "¿campeón en top-N?" es robusto.
- Elo de participantes congelado al inicio del torneo (no se actualiza partido a partido) — simplificación documentada, refinable (§10).

**Tech Stack:** Reusa `GBMPoissonModel`, `EloBaseline`, `attach_pre_match_elo` (F4), `simulate_tournament` + `RateProvider` (F5), métricas (F3). NumPy, Pandas, SQLite, pytest.

**Fechas de inicio y campeones (verificación):** 2010 Sudáfrica (2010-06-11, España), 2014 Brasil (2014-06-12, Alemania), 2018 Rusia (2018-06-14, Francia), 2022 Qatar (2022-11-20, Argentina).

---

### Task 1: Utilidades históricas (`src/evaluation/historical.py`)

**Files:** Create `src/evaluation/historical.py`, `tests/test_historical.py`

- [ ] **Step 1.1: Test que falla** — `tests/test_historical.py`

```python
import pandas as pd

from src.evaluation.historical import extract_groups


def _synthetic_group_stage():
    """2 grupos de 3 equipos (mini) que juegan todos contra todos."""
    rows = []
    g1 = ["A", "B", "C"]
    g2 = ["D", "E", "F"]
    import itertools
    for grp in (g1, g2):
        for h, a in itertools.combinations(grp, 2):
            rows.append({"home_team": h, "away_team": a, "home_goals": 1, "away_goals": 0})
    return pd.DataFrame(rows)


def test_extract_groups_finds_components():
    groups = extract_groups(_synthetic_group_stage(), group_size=3)
    # 2 grupos de 3
    assert len(groups) == 2
    sets = sorted([sorted(g) for g in groups.values()])
    assert sets == [["A", "B", "C"], ["D", "E", "F"]]
```

- [ ] **Step 1.2:** `pytest tests/test_historical.py -v` → FAIL

- [ ] **Step 1.3: Implementación** — `src/evaluation/historical.py`

```python
"""Utilidades para validacion historica sobre mundiales pasados."""
import sqlite3

import pandas as pd

# Mundiales objetivo: año -> (fecha de inicio, campeon real)
WORLD_CUPS = {
    2010: ("2010-06-11", "Spain"),
    2014: ("2014-06-12", "Germany"),
    2018: ("2018-06-14", "France"),
    2022: ("2022-11-20", "Argentina"),
}


def wc_matches(con: sqlite3.Connection, year: int) -> pd.DataFrame:
    """Partidos del Mundial 'year' con nombres de equipos y resultado."""
    df = pd.read_sql(
        """SELECT m.date, t1.name AS home_team, t2.name AS away_team,
                  m.home_goals, m.away_goals, m.neutral,
                  ts.name AS shootout_winner
           FROM matches m
           JOIN teams t1 ON t1.team_id = m.home_team_id
           JOIN teams t2 ON t2.team_id = m.away_team_id
           LEFT JOIN teams ts ON ts.team_id = m.shootout_winner_id
           WHERE m.tournament = 'FIFA World Cup'
             AND substr(m.date, 1, 4) = ?
           ORDER BY m.date""",
        con, params=[str(year)], parse_dates=["date"])
    return df


def extract_groups(group_stage: pd.DataFrame, group_size: int = 4) -> dict:
    """Reconstruye grupos como componentes conexas del grafo 'jugaron entre si'.
    Espera SOLO partidos de fase de grupos (cada equipo juega group_size-1)."""
    adj: dict[str, set] = {}
    for m in group_stage.itertuples(index=False):
        adj.setdefault(m.home_team, set()).add(m.away_team)
        adj.setdefault(m.away_team, set()).add(m.home_team)

    seen = set()
    groups = {}
    gid = 0
    for team in adj:
        if team in seen:
            continue
        # BFS componente
        comp = set()
        stack = [team]
        while stack:
            t = stack.pop()
            if t in comp:
                continue
            comp.add(t)
            stack.extend(adj[t] - comp)
        seen |= comp
        groups[f"G{gid + 1}"] = sorted(comp)
        gid += 1
    return groups


def extract_field(con: sqlite3.Connection, year: int) -> dict:
    """Grupos reales (8 de 4) reconstruidos desde los primeros 48 partidos."""
    m = wc_matches(con, year)
    n_group_matches = 48  # formato 32 equipos: 8 grupos x 6 partidos
    group_stage = m.head(n_group_matches)
    return extract_groups(group_stage, group_size=4)


def champion_of(con: sqlite3.Connection, year: int) -> str:
    """Ganador de la final (ultimo partido del torneo; penales si empate)."""
    m = wc_matches(con, year)
    final = m.iloc[-1]
    if final["home_goals"] > final["away_goals"]:
        return final["home_team"]
    if final["away_goals"] > final["home_goals"]:
        return final["away_team"]
    return final["shootout_winner"]


def elos_before(con: sqlite3.Connection, date: str, teams: list[str]) -> dict:
    """Elo de cada equipo justo antes de 'date' (point-in-time)."""
    elo = pd.read_sql(
        """SELECT t.name AS team, e.elo, e.date FROM elo_history e
           JOIN teams t ON t.team_id = e.team_id
           WHERE e.date < ? AND t.name IN ({})
           ORDER BY e.date""".format(",".join(["?"] * len(teams))),
        con, params=[date] + teams)
    latest = elo.groupby("team")["elo"].last().to_dict()
    return {t: float(latest.get(t, 1500.0)) for t in teams}
```

- [ ] **Step 1.4:** `pytest tests/test_historical.py -v` → PASS
- [ ] **Step 1.5: Verificación real** — script rápido que imprima `extract_field(con, 2018)` y `champion_of(con, 2018)`; confirmar 8 grupos de 4 y campeón = France.
- [ ] **Step 1.6: Commit** — `git commit -m "feat: utilidades de validacion historica (grupos, campeon, elo pre-torneo)"`

---

### Task 2: Backtest del torneo (`src/evaluation/backtest_tournament.py`)

**Files:** Create `src/evaluation/backtest_tournament.py`

- [ ] **Step 2.1: Implementación** — `src/evaluation/backtest_tournament.py`

```python
"""Backtest del sistema completo contra mundiales pasados (2010-2022)."""
import sqlite3

import numpy as np
import pandas as pd

from src.config import DB_PATH, REPORTS_DIR
from src.evaluation.historical import (WORLD_CUPS, champion_of, elos_before,
                                       extract_field, wc_matches)
from src.evaluation.metrics import accuracy_1x2, log_loss_1x2, rps
from src.features.elo_features import attach_pre_match_elo, tournament_importance
from src.models.baseline_elo import EloBaseline
from src.models.gbm_poisson import GBMPoissonModel
from src.simulation.monte_carlo import simulate_tournament
from src.simulation.rates import RateProvider


def _outcome(hg, ag):
    return 0 if hg > ag else (1 if hg == ag else 2)


def _train_models(con, before_date):
    """Entrena GBM-Poisson y baseline Elo con partidos anteriores a before_date."""
    matches = pd.read_sql(
        """SELECT m.date, t1.name AS home_team, t2.name AS away_team,
                  m.home_goals, m.away_goals, m.neutral, m.tournament
           FROM matches m JOIN teams t1 ON t1.team_id=m.home_team_id
           JOIN teams t2 ON t2.team_id=m.away_team_id
           WHERE m.date >= '2002-01-01' AND m.date < ? ORDER BY m.date""",
        con, params=[before_date], parse_dates=["date"])
    elo = pd.read_sql("SELECT t.name AS team, e.date, e.elo FROM elo_history e "
                      "JOIN teams t ON t.team_id=e.team_id WHERE e.date < ?",
                      con, params=[before_date], parse_dates=["date"])
    matches = attach_pre_match_elo(matches, elo)
    gbm = GBMPoissonModel().fit(matches)

    base_df = matches.assign(outcome=[_outcome(h, a) for h, a in
                                      zip(matches["home_goals"], matches["away_goals"])])
    baseline = EloBaseline().fit(base_df[["elo_diff", "outcome"]])
    return gbm, baseline


def _match_level_eval(con, year, start_date, gbm, baseline):
    """Predice cada partido real del Mundial con el modelo pre-torneo."""
    m = wc_matches(con, year)
    teams = sorted(set(m["home_team"]) | set(m["away_team"]))
    elos = elos_before(con, start_date, teams)
    imp = tournament_importance("FIFA World Cup")

    actual = np.array([_outcome(r.home_goals, r.away_goals) for r in m.itertuples()])
    gbm_probs = np.array([
        gbm.predict_proba_1x2(elos[r.home_team], elos[r.away_team], imp, bool(r.neutral))
        for r in m.itertuples()])
    base_probs = baseline.predict_proba(
        np.array([elos[r.home_team] - elos[r.away_team] for r in m.itertuples()]))
    return {
        "n_matches": len(m),
        "gbm_rps": round(rps(gbm_probs, actual), 4),
        "base_rps": round(rps(base_probs, actual), 4),
        "gbm_logloss": round(log_loss_1x2(gbm_probs, actual), 4),
        "gbm_acc": round(accuracy_1x2(gbm_probs, actual), 4),
    }


def _tournament_eval(con, year, start_date, gbm, n_sims, champion):
    """Simula el torneo con grupos reales y evalua el ranking de P(campeon)."""
    groups = extract_field(con, year)
    teams = [t for g in groups.values() for t in g]
    elos = elos_before(con, start_date, teams)
    rp = RateProvider(gbm, elos)
    res = simulate_tournament(groups, rp, n_sims=n_sims,
                              n_qualify_per_group=2, n_best_thirds=0, seed=year)
    res = res.reset_index(drop=True)
    rank = res.index[res["team"] == champion]
    champ_rank = int(rank[0]) + 1 if len(rank) else -1
    champ_prob = float(res.loc[res["team"] == champion, "p_champion"].iloc[0]) if champ_rank > 0 else 0.0
    return {
        "predicted_champion": res.iloc[0]["team"],
        "predicted_prob": round(float(res.iloc[0]["p_champion"]) * 100, 1),
        "real_champion": champion,
        "real_champ_rank": champ_rank,
        "real_champ_prob": round(champ_prob * 100, 1),
        "n_groups": len(groups),
    }


def run(n_sims: int = 20000):
    con = sqlite3.connect(DB_PATH)
    rows_match, rows_tourn = [], []
    for year, (start, champion) in WORLD_CUPS.items():
        gbm, baseline = _train_models(con, start)
        me = _match_level_eval(con, year, start, gbm, baseline)
        te = _tournament_eval(con, year, start, gbm, n_sims, champion)
        rows_match.append({"year": year, **me})
        rows_tourn.append({"year": year, **te})
    con.close()
    return pd.DataFrame(rows_match), pd.DataFrame(rows_tourn)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20000)
    args = parser.parse_args()

    match_df, tourn_df = run(n_sims=args.n)

    # gate
    wins = int((match_df["gbm_rps"] < match_df["base_rps"]).sum())
    top5 = int(((tourn_df["real_champ_rank"] >= 1) & (tourn_df["real_champ_rank"] <= 5)).sum())

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "f6_historical_validation.md"
    lines = [
        "# Validacion historica (Mundiales 2010, 2014, 2018, 2022)",
        "", f"Simulaciones por torneo: {args.n:,}. Entrenamiento con corte temporal",
        "estricto (solo datos anteriores al inicio de cada Mundial).", "",
        "## Nivel partido (GBM-Poisson vs baseline Elo)",
        "", match_df.to_markdown(index=False), "",
        f"GBM gana en RPS en **{wins}/4** mundiales.", "",
        "## Nivel torneo (ranking de P(campeon))",
        "", tourn_df.to_markdown(index=False), "",
        f"Campeon real en **top-5** de P(campeon) en **{top5}/4** mundiales.", "",
        "## Gate F6",
        f"- (a) GBM supera baseline en RPS en >=3/4: {'PASS' if wins >= 3 else 'FAIL'} ({wins}/4)",
        f"- (b) Campeon real en top-5 en >=3/4: {'PASS' if top5 >= 3 else 'FAIL'} ({top5}/4)",
        "", "Nota: bracket por siembra y Elo congelado al inicio (ver plan). El campo y",
        "grupos son los REALES de cada torneo (reconstruidos desde la base de datos).",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2.2: Corrida real** — `python -m src.evaluation.backtest_tournament --n 20000`
(Si tarda demasiado para primer plano, ejecutar en segundo plano.) Verificar gate.
- [ ] **Step 2.3: Commit** — `git add -f outputs/reports/f6_historical_validation.md && git add src/evaluation/backtest_tournament.py && git commit -m "feat: backtest historico del sistema completo (2010-2022)"`

---

### Task 3: Validación y cierre F6

- [ ] **Step 3.1:** `pytest -v` → todos PASS
- [ ] **Step 3.2:** Revisar el reporte: documentar honestamente el resultado del gate (si el campeón real no queda en top-5 en algún torneo, explicar — p.ej. 2010 España era favorita, pero sorpresas pasan). El valor es el aprendizaje, no un número perfecto.
- [ ] **Step 3.3:** Actualizar README (marcar Fase 6 con resultados reales).
- [ ] **Step 3.4: Commit + push** — `git commit -m "docs: fase 6 completada"; git push`

---

## Self-Review

- **Cobertura del spec F6 (§7, §8):** entrenar con corte temporal estricto ✅, predecir partido a partido ✅, simular torneo completo ✅, comparar vs campeón real ✅, vs baseline Elo ✅, los 4 mundiales ✅, informe por torneo reproducible ✅.
- **Reuso:** `simulate_tournament`/`RateProvider` (F5), `GBMPoissonModel`/`attach_pre_match_elo`/`EloBaseline` (F4/F3), métricas (F3) — sin reimplementar.
- **Reconstrucción de grupos:** desde los primeros 48 partidos del torneo como componentes conexas; testeado en caso sintético; verificado en 2018 (debe dar 8 grupos de 4, campeón France).
- **Desviaciones documentadas:** bracket por siembra; Elo congelado al inicio del torneo.
- **Gate honesto:** se reporta PASS/FAIL real; si falla algún criterio, se explica como aprendizaje, no se ajusta el número.
- **Sin placeholders:** código completo en cada step.
