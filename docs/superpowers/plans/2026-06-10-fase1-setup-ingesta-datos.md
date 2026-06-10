# Fase 1 — Setup e Ingesta de Datos: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repo funcional con ETL de partidos internacionales (GitHub martj42), Elo propio calculado, rankings FIFA, sedes 2026 y base SQLite+Parquet, cumpliendo los criterios de aceptación de la Fase 1 del DOCUMENTO_TECNICO.md (§8).

**Architecture:** Pipeline ETL en módulos pequeños (`src/etl/download.py → clean.py → load.py`), Elo en `src/features/elo.py`, orquestación en `src/etl/run_all.py`, validación de aceptación en `src/etl/validate.py`. SQLite como base relacional + Parquet para análisis. Datos crudos fechados en `data/raw/`.

**Tech Stack:** Python 3.11+, Pandas, NumPy, PyArrow, Requests, PyYAML, kagglehub (rankings FIFA), pytest, SQLite (stdlib `sqlite3`), Git.

**Decisiones clave (desviaciones documentadas del DOCUMENTO_TECNICO):**
- Fuente principal de partidos: **GitHub `martj42/international_results` (raw CSV)** en lugar del espejo Kaggle — mismo autor y datos, sin necesidad de credenciales Kaggle. Kaggle queda como plan B.
- Rankings FIFA vía `kagglehub` (descarga anónima de datasets públicos); si falla, el pipeline continúa con warning (la tabla `fifa_rankings` queda vacía y se reporta).
- Sin Makefile (Windows): puntos de entrada `python -m src.etl.run_all` y `python -m src.etl.validate`.
- Elo propio se calcula desde 1872 (converge mucho antes del 2000); la correlación contra eloratings.net se intenta automáticamente vía `World.tsv` y, si el formato/red falla, se degrada a un chequeo de sanidad del top-10 documentado en el reporte.

**Directorio raíz del proyecto:** `C:\Users\cesar\Documents\Personal\Proyectos\mundial-2026-ml`

---

### Task 0: Verificar herramientas (Python, Git)

- [ ] **Step 0.1:** Ejecutar `python --version` (o `py -3 --version`) y `git --version`. Ambos deben existir. Si Python no existe, instalar con `winget install Python.Python.3.12` y reabrir la sesión. Si Git no existe, `winget install Git.Git`.

---

### Task 1: Scaffold del repo

**Files:**
- Create: `.gitignore`, `pyproject.toml`, `src/__init__.py`, `src/etl/__init__.py`, `src/features/__init__.py`, `tests/__init__.py`, `src/config.py`

- [ ] **Step 1.1: git init** (el folder ya tiene README.md y docs/)

```
git init
git add README.md docs/
git commit -m "docs: documento técnico de diseño v1.0"
```

- [ ] **Step 1.2: Crear `.gitignore`**

```gitignore
__pycache__/
*.pyc
.venv/
db/*.sqlite
data/raw/
data/external/
data/processed/
data/features/
outputs/simulations/
.pytest_cache/
*.egg-info/
```

- [ ] **Step 1.3: Crear `pyproject.toml`**

```toml
[project]
name = "mundial-2026-ml"
version = "0.1.0"
description = "Prediccion del Mundial FIFA 2026 con ML - proyecto educativo"
requires-python = ">=3.11"
dependencies = [
    "pandas>=2.0",
    "numpy>=1.26",
    "pyarrow>=15",
    "requests>=2.31",
    "pyyaml>=6.0",
    "kagglehub>=0.3",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["src*"]
```

- [ ] **Step 1.4: Crear `src/config.py`**

```python
"""Rutas centrales del proyecto."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
EXTERNAL_DIR = DATA_DIR / "external"
PROCESSED_DIR = DATA_DIR / "processed"
DB_PATH = PROJECT_ROOT / "db" / "worldcup.sqlite"
CONFIGS_DIR = PROJECT_ROOT / "configs"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
```

- [ ] **Step 1.5:** Crear paquetes vacíos: `src/__init__.py`, `src/etl/__init__.py`, `src/features/__init__.py`, `tests/__init__.py` (archivos vacíos).

- [ ] **Step 1.6: Crear venv e instalar dependencias**

```
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
```

Expected: instalación sin errores. Verificar: `.venv\Scripts\python -c "import pandas, numpy, pyarrow, requests, yaml; print('ok')"` → `ok`.

- [ ] **Step 1.7: Commit**

```
git add .gitignore pyproject.toml src/ tests/
git commit -m "feat: scaffold del repo (pyproject, config, paquetes)"
```

---

### Task 2: Descarga de datos crudos (`src/etl/download.py`)

**Files:**
- Create: `src/etl/download.py`
- Test: `tests/test_download.py`

- [ ] **Step 2.1: Test que falla** — `tests/test_download.py`

```python
import datetime as dt
from src.etl.download import save_raw, SOURCES


def test_sources_define_the_three_files():
    assert set(SOURCES) == {"results", "shootouts", "former_names"}


def test_save_raw_writes_dated_and_latest_copies(tmp_path):
    content = b"date,home_team\n2022-12-18,Argentina\n"
    dated = save_raw("results", content, raw_dir=tmp_path, today=dt.date(2026, 6, 10))
    assert dated.name == "results_2026-06-10.csv"
    assert dated.read_bytes() == content
    assert (tmp_path / "results_latest.csv").read_bytes() == content
```

- [ ] **Step 2.2:** `pytest tests/test_download.py -v` → FAIL (ModuleNotFoundError).

- [ ] **Step 2.3: Implementación** — `src/etl/download.py`

```python
"""Descarga de fuentes publicas con copia cruda fechada."""
import datetime as dt
from pathlib import Path

import requests

from src.config import RAW_DIR

_BASE = "https://raw.githubusercontent.com/martj42/international_results/master"
SOURCES = {
    "results": f"{_BASE}/results.csv",
    "shootouts": f"{_BASE}/shootouts.csv",
    "former_names": f"{_BASE}/former_names.csv",
}


def fetch(url: str, timeout: int = 60) -> bytes:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.content


def save_raw(name: str, content: bytes, raw_dir: Path = RAW_DIR,
             today: dt.date | None = None) -> Path:
    today = today or dt.date.today()
    raw_dir.mkdir(parents=True, exist_ok=True)
    dated = raw_dir / f"{name}_{today.isoformat()}.csv"
    dated.write_bytes(content)
    (raw_dir / f"{name}_latest.csv").write_bytes(content)
    return dated


def download_all() -> dict[str, Path]:
    return {name: save_raw(name, fetch(url)) for name, url in SOURCES.items()}


if __name__ == "__main__":
    for name, path in download_all().items():
        print(f"{name}: {path}")
```

- [ ] **Step 2.4:** `pytest tests/test_download.py -v` → PASS (2 tests).

- [ ] **Step 2.5: Descarga real** — `python -m src.etl.download`
Expected: imprime 3 rutas; `data/raw/results_latest.csv` > 2 MB (~48k filas).

- [ ] **Step 2.6: Commit** — `git add src/etl/download.py tests/test_download.py && git commit -m "feat: descarga de resultados internacionales (martj42)"`

---

### Task 3: Limpieza y normalización de nombres (`src/etl/clean.py`)

**Files:**
- Create: `src/etl/clean.py`
- Test: `tests/test_clean.py`

- [ ] **Step 3.1: Test que falla** — `tests/test_clean.py`

```python
import pandas as pd
from src.etl.clean import build_alias_map, clean_results


def _former_names():
    return pd.DataFrame({
        "current": ["DR Congo", "Indonesia"],
        "former": ["Zaïre", "Dutch East Indies"],
    })


def _raw_results():
    return pd.DataFrame({
        "date": ["2022-12-18", "2022-12-18", "1974-09-22", "bad-date"],
        "home_team": ["Argentina", "Argentina", "Zaïre", "Spain"],
        "away_team": ["France", "France", "Ghana", "France"],
        "home_score": [3, 3, 2, 1],
        "away_score": [3, 3, 1, 0],
        "tournament": ["FIFA World Cup"] * 3 + ["Friendly"],
        "city": ["Lusail", "Lusail", "Kinshasa", "Madrid"],
        "country": ["Qatar", "Qatar", "Zaire", "Spain"],
        "neutral": [True, True, False, False],
    })


def test_alias_map_maps_former_to_current():
    assert build_alias_map(_former_names())["Zaïre"] == "DR Congo"


def test_clean_results_dedups_normalizes_and_drops_bad_rows():
    out = clean_results(_raw_results(), build_alias_map(_former_names()))
    assert len(out) == 2                      # dedup + fila con fecha invalida fuera
    assert "Zaïre" not in set(out["home_team"])
    assert "DR Congo" in set(out["home_team"])
    assert out["date"].is_monotonic_increasing
    assert out["home_score"].dtype.kind == "i"
    assert out["neutral"].dtype == bool
```

- [ ] **Step 3.2:** `pytest tests/test_clean.py -v` → FAIL.

- [ ] **Step 3.3: Implementación** — `src/etl/clean.py`

```python
"""Limpieza y normalizacion de nombres de selecciones."""
import pandas as pd


def build_alias_map(former_names: pd.DataFrame) -> dict[str, str]:
    """former -> current, segun former_names.csv del dataset martj42."""
    return dict(zip(former_names["former"], former_names["current"]))


def normalize_teams(df: pd.DataFrame, alias_map: dict[str, str]) -> pd.DataFrame:
    out = df.copy()
    for col in ("home_team", "away_team"):
        out[col] = out[col].replace(alias_map)
    return out


def clean_results(raw: pd.DataFrame, alias_map: dict[str, str]) -> pd.DataFrame:
    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_team", "away_team",
                           "home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df["neutral"] = df["neutral"].astype(bool)
    df = normalize_teams(df, alias_map)
    df = df.drop_duplicates(subset=["date", "home_team", "away_team"])
    return df.sort_values("date").reset_index(drop=True)


def clean_shootouts(raw: pd.DataFrame, alias_map: dict[str, str]) -> pd.DataFrame:
    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_team", "away_team", "winner"])
    for col in ("home_team", "away_team", "winner"):
        df[col] = df[col].replace(alias_map)
    return df.drop_duplicates(subset=["date", "home_team", "away_team"]).reset_index(drop=True)
```

- [ ] **Step 3.4:** `pytest tests/test_clean.py -v` → PASS.
- [ ] **Step 3.5: Commit** — `git commit -m "feat: limpieza y normalizacion de nombres (former_names)"`

---

### Task 4: Carga a SQLite + Parquet (`src/etl/load.py`)

**Files:**
- Create: `src/etl/load.py`
- Test: `tests/test_load.py`

- [ ] **Step 4.1: Test que falla** — `tests/test_load.py`

```python
import sqlite3

import pandas as pd

from src.etl.load import load_matches


def _matches():
    return pd.DataFrame({
        "date": pd.to_datetime(["2022-12-18", "2022-12-09"]),
        "home_team": ["Argentina", "Netherlands"],
        "away_team": ["France", "Argentina"],
        "home_score": [3, 2],
        "away_score": [3, 2],
        "tournament": ["FIFA World Cup", "FIFA World Cup"],
        "city": ["Lusail", "Lusail"],
        "country": ["Qatar", "Qatar"],
        "neutral": [True, True],
    })


def _shootouts():
    return pd.DataFrame({
        "date": pd.to_datetime(["2022-12-18", "2022-12-09"]),
        "home_team": ["Argentina", "Netherlands"],
        "away_team": ["France", "Argentina"],
        "winner": ["Argentina", "Argentina"],
    })


def test_load_matches_builds_teams_and_links_shootouts(tmp_path):
    db = tmp_path / "test.sqlite"
    n = load_matches(_matches(), _shootouts(), db_path=db, processed_dir=tmp_path)
    assert n == 2
    con = sqlite3.connect(db)
    teams = pd.read_sql("SELECT * FROM teams", con)
    assert set(teams["name"]) == {"Argentina", "France", "Netherlands"}
    m = pd.read_sql("""
        SELECT m.*, t.name AS winner_name FROM matches m
        LEFT JOIN teams t ON t.team_id = m.shootout_winner_id
    """, con)
    con.close()
    assert set(m["winner_name"]) == {"Argentina"}
    assert (tmp_path / "matches.parquet").exists()
```

- [ ] **Step 4.2:** `pytest tests/test_load.py -v` → FAIL.

- [ ] **Step 4.3: Implementación** — `src/etl/load.py`

```python
"""Carga de tablas limpias a SQLite y Parquet."""
import sqlite3
from pathlib import Path

import pandas as pd

from src.config import DB_PATH, PROCESSED_DIR

SCHEMA = """
CREATE TABLE IF NOT EXISTS teams (
    team_id INTEGER PRIMARY KEY,
    name    TEXT NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS matches (
    match_id            INTEGER PRIMARY KEY,
    date                TEXT NOT NULL,
    tournament          TEXT NOT NULL,
    home_team_id        INTEGER NOT NULL REFERENCES teams(team_id),
    away_team_id        INTEGER NOT NULL REFERENCES teams(team_id),
    home_goals          INTEGER NOT NULL,
    away_goals          INTEGER NOT NULL,
    city                TEXT,
    country             TEXT,
    neutral             INTEGER NOT NULL,
    shootout_winner_id  INTEGER REFERENCES teams(team_id),
    UNIQUE (date, home_team_id, away_team_id)
);
CREATE TABLE IF NOT EXISTS elo_history (
    team_id  INTEGER NOT NULL REFERENCES teams(team_id),
    date     TEXT NOT NULL,
    match_id INTEGER REFERENCES matches(match_id),
    elo      REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS fifa_rankings (
    team_id      INTEGER REFERENCES teams(team_id),
    team_name    TEXT NOT NULL,
    ranking_date TEXT NOT NULL,
    rank         INTEGER NOT NULL,
    points       REAL
);
"""


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    con.executescript(SCHEMA)
    return con


def load_matches(matches: pd.DataFrame, shootouts: pd.DataFrame,
                 db_path: Path = DB_PATH,
                 processed_dir: Path = PROCESSED_DIR) -> int:
    """Reconstruye teams y matches desde cero (carga idempotente)."""
    con = _connect(db_path)
    try:
        con.execute("DELETE FROM matches")
        con.execute("DELETE FROM teams")

        names = sorted(set(matches["home_team"]) | set(matches["away_team"])
                       | set(shootouts["winner"]))
        teams = pd.DataFrame({"team_id": range(1, len(names) + 1), "name": names})
        teams.to_sql("teams", con, if_exists="append", index=False)
        tid = dict(zip(teams["name"], teams["team_id"]))

        df = matches.copy()
        df["home_team_id"] = df["home_team"].map(tid)
        df["away_team_id"] = df["away_team"].map(tid)
        key = ["date", "home_team", "away_team"]
        so = shootouts.set_index(key)["winner"]
        df["shootout_winner_id"] = (
            df.set_index(key).index.map(so).map(tid).astype("Int64").to_numpy()
        )
        df["neutral"] = df["neutral"].astype(int)
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        out = df[["date", "tournament", "home_team_id", "away_team_id",
                  "home_score", "away_score", "city", "country", "neutral",
                  "shootout_winner_id"]].rename(
            columns={"home_score": "home_goals", "away_score": "away_goals"})
        out.to_sql("matches", con, if_exists="append", index=False)
        con.commit()

        processed_dir.mkdir(parents=True, exist_ok=True)
        matches.to_parquet(processed_dir / "matches.parquet", index=False)
        teams.to_parquet(processed_dir / "teams.parquet", index=False)
        return len(out)
    finally:
        con.close()
```

- [ ] **Step 4.4:** `pytest tests/test_load.py -v` → PASS.
- [ ] **Step 4.5: Commit** — `git commit -m "feat: carga a SQLite + Parquet (teams, matches, shootouts)"`

---

### Task 5: Elo propio (`src/features/elo.py`)

**Files:**
- Create: `src/features/elo.py`
- Test: `tests/test_elo.py`

Fórmula (World Football Elo Ratings, eloratings.net):
`R' = R + K·G·(W − We)`, con `We = 1/(1+10^(−dr/400))`, `dr = elo_home + 100·(no neutral) − elo_away`.
K: Mundial 60; finales continentales/Confederaciones 50; eliminatorias 40; otros torneos 30; amistosos 20.
G: dif 0–1 → 1.0; dif 2 → 1.5; dif 3 → 1.75; dif ≥4 → 1.75 + (dif−3)/8.

- [ ] **Step 5.1: Test que falla** — `tests/test_elo.py`

```python
import pandas as pd
import pytest

from src.features.elo import compute_elo, expected_score, goal_multiplier, k_factor


def test_k_factor():
    assert k_factor("FIFA World Cup") == 60
    assert k_factor("Copa América") == 50
    assert k_factor("UEFA Euro") == 50
    assert k_factor("FIFA World Cup qualification") == 40
    assert k_factor("UEFA Nations League") == 30
    assert k_factor("Friendly") == 20


def test_goal_multiplier():
    assert goal_multiplier(0) == 1.0
    assert goal_multiplier(1) == 1.0
    assert goal_multiplier(2) == 1.5
    assert goal_multiplier(3) == 1.75
    assert goal_multiplier(5) == pytest.approx(2.0)


def test_expected_score_neutral_equal_ratings():
    assert expected_score(1500, 1500, neutral=True) == pytest.approx(0.5)


def test_expected_score_home_advantage():
    assert expected_score(1500, 1500, neutral=False) == pytest.approx(0.640065, abs=1e-5)


def _one_match(neutral, home_score=1, away_score=0, tournament="Friendly"):
    return pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01"]),
        "home_team": ["A"], "away_team": ["B"],
        "home_score": [home_score], "away_score": [away_score],
        "tournament": [tournament], "neutral": [neutral],
    })


def test_compute_elo_hand_calculated_neutral_friendly():
    history, ratings = compute_elo(_one_match(neutral=True))
    # We=0.5, K=20, G=1.0 -> delta = 20*1*(1-0.5) = 10
    assert ratings["A"] == pytest.approx(1510.0)
    assert ratings["B"] == pytest.approx(1490.0)
    assert len(history) == 2  # una fila por equipo


def test_compute_elo_hand_calculated_home_win_by_two():
    _, ratings = compute_elo(_one_match(neutral=False, home_score=2))
    # We=0.640065, K=20, G=1.5 -> delta = 30*(1-0.640065) = 10.798
    assert ratings["A"] == pytest.approx(1510.798, abs=1e-3)
    assert ratings["B"] == pytest.approx(1489.202, abs=1e-3)
```

- [ ] **Step 5.2:** `pytest tests/test_elo.py -v` → FAIL.

- [ ] **Step 5.3: Implementación** — `src/features/elo.py`

```python
"""Elo propio segun la formula de eloratings.net (World Football Elo)."""
import pandas as pd

INITIAL_ELO = 1500.0
HOME_ADVANTAGE = 100.0

_CONTINENTAL_FINALS = {
    "UEFA Euro", "Copa América", "African Cup of Nations", "AFC Asian Cup",
    "CONCACAF Championship", "Gold Cup", "Oceania Nations Cup",
    "Confederations Cup", "King's Cup",  # King's Cup se ignora: ver k_factor
}
_CONTINENTAL_FINALS.discard("King's Cup")


def k_factor(tournament: str) -> int:
    if tournament == "FIFA World Cup":
        return 60
    if tournament in _CONTINENTAL_FINALS:
        return 50
    if "qualification" in tournament.lower():
        return 40
    if tournament == "Friendly":
        return 20
    return 30


def goal_multiplier(goal_diff: int) -> float:
    d = abs(goal_diff)
    if d <= 1:
        return 1.0
    if d == 2:
        return 1.5
    if d == 3:
        return 1.75
    return 1.75 + (d - 3) / 8


def expected_score(elo_home: float, elo_away: float, neutral: bool) -> float:
    dr = elo_home + (0 if neutral else HOME_ADVANTAGE) - elo_away
    return 1.0 / (1.0 + 10 ** (-dr / 400.0))


def compute_elo(matches: pd.DataFrame,
                initial: float = INITIAL_ELO) -> tuple[pd.DataFrame, dict[str, float]]:
    """Recorre los partidos en orden cronologico y devuelve
    (historial post-partido por equipo, ratings finales)."""
    matches = matches.sort_values("date")
    ratings: dict[str, float] = {}
    rows: list[dict] = []
    for m in matches.itertuples(index=False):
        rh = ratings.get(m.home_team, initial)
        ra = ratings.get(m.away_team, initial)
        we = expected_score(rh, ra, bool(m.neutral))
        if m.home_score > m.away_score:
            w = 1.0
        elif m.home_score < m.away_score:
            w = 0.0
        else:
            w = 0.5
        delta = (k_factor(m.tournament)
                 * goal_multiplier(m.home_score - m.away_score)
                 * (w - we))
        ratings[m.home_team] = rh + delta
        ratings[m.away_team] = ra - delta
        rows.append({"team": m.home_team, "date": m.date, "elo": ratings[m.home_team]})
        rows.append({"team": m.away_team, "date": m.date, "elo": ratings[m.away_team]})
    return pd.DataFrame(rows), ratings
```

- [ ] **Step 5.4:** `pytest tests/test_elo.py -v` → PASS (6 tests).
- [ ] **Step 5.5: Commit** — `git commit -m "feat: calculo de Elo propio (formula eloratings.net)"`

---

### Task 6: Rankings FIFA (`src/etl/fifa_rankings.py`)

**Files:**
- Create: `src/etl/fifa_rankings.py`
- Test: `tests/test_fifa_rankings.py`

- [ ] **Step 6.1: Test que falla** — `tests/test_fifa_rankings.py` (solo la transformación; la descarga real se prueba en la corrida E2E)

```python
import pandas as pd

from src.etl.fifa_rankings import tidy_rankings


def test_tidy_rankings_normalizes_columns_and_names():
    raw = pd.DataFrame({
        "rank": [1, 2],
        "country_full": ["Argentina", "IR Iran"],
        "total_points": [1860.1, 1600.0],
        "rank_date": ["2024-04-04", "2024-04-04"],
    })
    out = tidy_rankings(raw, alias_map={"IR Iran": "Iran"})
    assert list(out.columns) == ["team_name", "ranking_date", "rank", "points"]
    assert set(out["team_name"]) == {"Argentina", "Iran"}
```

- [ ] **Step 6.2:** `pytest tests/test_fifa_rankings.py -v` → FAIL.

- [ ] **Step 6.3: Implementación** — `src/etl/fifa_rankings.py`

```python
"""Rankings FIFA historicos via kagglehub (dataset publico, sin credenciales)."""
from pathlib import Path

import pandas as pd

KAGGLE_DATASET = "cashncarry/fifaworldranking"


def download_rankings() -> pd.DataFrame | None:
    """Devuelve el CSV crudo del dataset, o None si no es posible descargar."""
    try:
        import kagglehub
        path = Path(kagglehub.dataset_download(KAGGLE_DATASET))
        csvs = sorted(path.rglob("*.csv"))
        if not csvs:
            return None
        return pd.read_csv(csvs[0])
    except Exception as exc:  # red, cuota, dataset movido: el pipeline sigue
        print(f"WARN: rankings FIFA no disponibles ({exc})")
        return None


def tidy_rankings(raw: pd.DataFrame, alias_map: dict[str, str]) -> pd.DataFrame:
    df = raw.rename(columns={"country_full": "team_name",
                             "rank_date": "ranking_date",
                             "total_points": "points"})
    df["team_name"] = df["team_name"].replace(alias_map)
    df["ranking_date"] = pd.to_datetime(df["ranking_date"]).dt.strftime("%Y-%m-%d")
    return df[["team_name", "ranking_date", "rank", "points"]]
```

Nota: nombres FIFA difieren de martj42 ("IR Iran", "USA", "Korea Republic"…). El alias map para rankings se define en `run_all.py` (Task 7) combinando `former_names` + `FIFA_ALIASES` manual.

- [ ] **Step 6.4:** `pytest tests/test_fifa_rankings.py -v` → PASS.
- [ ] **Step 6.5: Commit** — `git commit -m "feat: ingesta de rankings FIFA via kagglehub"`

---

### Task 7: Orquestación E2E (`src/etl/run_all.py`)

**Files:**
- Create: `src/etl/run_all.py`

- [ ] **Step 7.1: Implementación** — `src/etl/run_all.py`

```python
"""Pipeline ETL completo: descarga -> limpieza -> carga -> Elo -> rankings."""
import sqlite3

import pandas as pd

from src.config import DB_PATH, PROCESSED_DIR, RAW_DIR
from src.etl.clean import build_alias_map, clean_results, clean_shootouts
from src.etl.download import download_all
from src.etl.fifa_rankings import download_rankings, tidy_rankings
from src.etl.load import load_matches
from src.features.elo import compute_elo

# Nombres del dataset FIFA -> nombres canonicos martj42
FIFA_ALIASES = {
    "IR Iran": "Iran", "Korea Republic": "South Korea",
    "Korea DPR": "North Korea", "USA": "United States",
    "Côte d'Ivoire": "Ivory Coast", "Cabo Verde": "Cape Verde",
    "China PR": "China", "Congo DR": "DR Congo",
    "Türkiye": "Turkey", "Czechia": "Czech Republic",
    "St. Kitts and Nevis": "Saint Kitts and Nevis",
    "St. Lucia": "Saint Lucia", "St. Vincent / Grenadines": "Saint Vincent and the Grenadines",
    "Brunei Darussalam": "Brunei", "Kyrgyz Republic": "Kyrgyzstan",
    "Hong Kong, China": "Hong Kong", "Macau, China": "Macau",
    "Chinese Taipei": "Taiwan", "North Macedonia": "North Macedonia",
}


def main() -> None:
    print("1/5 Descargando fuentes...")
    download_all()

    print("2/5 Limpiando...")
    former = pd.read_csv(RAW_DIR / "former_names_latest.csv")
    alias = build_alias_map(former)
    matches = clean_results(pd.read_csv(RAW_DIR / "results_latest.csv"), alias)
    shootouts = clean_shootouts(pd.read_csv(RAW_DIR / "shootouts_latest.csv"), alias)

    print("3/5 Cargando a SQLite + Parquet...")
    n = load_matches(matches, shootouts)
    print(f"   {n} partidos cargados")

    print("4/5 Calculando Elo propio...")
    history, ratings = compute_elo(matches)
    history.to_parquet(PROCESSED_DIR / "elo_history.parquet", index=False)
    con = sqlite3.connect(DB_PATH)
    tid = dict(pd.read_sql("SELECT name, team_id FROM teams", con).values)
    h = history.assign(team_id=history["team"].map(tid),
                       date=history["date"].dt.strftime("%Y-%m-%d"))
    con.execute("DELETE FROM elo_history")
    h[["team_id", "date", "elo"]].to_sql("elo_history", con,
                                         if_exists="append", index=False)
    con.commit()

    print("5/5 Rankings FIFA...")
    raw_rank = download_rankings()
    if raw_rank is not None:
        ranks = tidy_rankings(raw_rank, alias_map={**alias, **FIFA_ALIASES})
        ranks = ranks.assign(team_id=ranks["team_name"].map(tid))
        con.execute("DELETE FROM fifa_rankings")
        ranks.to_sql("fifa_rankings", con, if_exists="append", index=False)
        con.commit()
        ranks.to_parquet(PROCESSED_DIR / "fifa_rankings.parquet", index=False)
        print(f"   {len(ranks)} filas de ranking")
    con.close()

    top = sorted(ratings.items(), key=lambda kv: -kv[1])[:10]
    print("\nTop-10 Elo propio:")
    for i, (team, elo) in enumerate(top, 1):
        print(f"  {i:2d}. {team:<15s} {elo:7.1f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 7.2: Corrida real** — `python -m src.etl.run_all`
Expected: 5 pasos sin excepción; >40 000 partidos cargados; top-10 Elo con potencias plausibles (Argentina, España, Francia, Brasil…).

- [ ] **Step 7.3: Commit** — `git commit -m "feat: pipeline ETL end-to-end (run_all)"`

---

### Task 8: Validación de aceptación F1 (`src/etl/validate.py`)

**Files:**
- Create: `src/etl/validate.py`
- Output: `outputs/reports/f1_validation.md`

Criterios del gate (DOCUMENTO_TECNICO §8 F1):
1. ≥99 % de partidos 2000–2026 cargados sin duplicados.
2. Conciliación de nombres: 0 equipos huérfanos (ningún nombre "former" sobrevive).
3. Elo propio correlaciona >0.95 con eloratings.net (vía `World.tsv`; si la red/formato falla, chequeo de sanidad top-10 y se reporta como verificación manual pendiente).

- [ ] **Step 8.1: Implementación** — `src/etl/validate.py`

```python
"""Reporte de validacion del gate F1."""
import datetime as dt
import sqlite3

import pandas as pd
import requests

from src.config import DB_PATH, RAW_DIR, REPORTS_DIR

ELO_SANITY_SET = {
    "Argentina", "France", "Spain", "Brazil", "England", "Portugal",
    "Netherlands", "Germany", "Italy", "Belgium", "Uruguay", "Colombia",
    "Croatia", "Morocco", "Japan",
}
# Codigos de eloratings.net para las selecciones grandes (suficiente para correlacion)
ELORATINGS_CODES = {
    "AR": "Argentina", "FR": "France", "ES": "Spain", "BR": "Brazil",
    "EN": "England", "PT": "Portugal", "NL": "Netherlands", "DE": "Germany",
    "IT": "Italy", "BE": "Belgium", "UY": "Uruguay", "CO": "Colombia",
    "HR": "Croatia", "MA": "Morocco", "JP": "Japan", "MX": "Mexico",
    "US": "United States", "DK": "Denmark", "CH": "Switzerland",
    "AT": "Austria", "EC": "Ecuador", "SN": "Senegal", "TR": "Turkey",
    "NO": "Norway", "SE": "Sweden", "PL": "Poland", "GR": "Greece",
    "RS": "Serbia", "AU": "Australia", "KR": "South Korea",
}


def fetch_eloratings() -> pd.DataFrame | None:
    """World.tsv de eloratings.net: sin header; col 2 = codigo, col 3 = rating."""
    try:
        resp = requests.get("https://www.eloratings.net/World.tsv", timeout=30,
                            headers={"User-Agent": "Mozilla/5.0 (educational project)"})
        resp.raise_for_status()
        df = pd.read_csv(__import__("io").StringIO(resp.text), sep="\t", header=None)
        out = df.iloc[:, [2, 3]].copy()
        out.columns = ["code", "elo_ref"]
        out["elo_ref"] = pd.to_numeric(out["elo_ref"], errors="coerce")
        out["team"] = out["code"].map(ELORATINGS_CODES)
        out = out.dropna(subset=["team", "elo_ref"])
        return out[["team", "elo_ref"]] if len(out) >= 15 else None
    except Exception as exc:
        print(f"WARN: eloratings.net no disponible ({exc})")
        return None


def main() -> None:
    con = sqlite3.connect(DB_PATH)
    matches = pd.read_sql("SELECT * FROM matches", con)
    teams = pd.read_sql("SELECT * FROM teams", con)
    elo = pd.read_sql("""
        SELECT t.name AS team, e.elo FROM elo_history e
        JOIN teams t ON t.team_id = e.team_id
        WHERE e.date = (SELECT MAX(date) FROM elo_history x WHERE x.team_id = e.team_id)
    """, con)
    n_rank = pd.read_sql("SELECT COUNT(*) AS n FROM fifa_rankings", con)["n"][0]
    con.close()

    lines = [f"# Validacion Fase 1 — {dt.date.today().isoformat()}", ""]
    ok = []

    # C1: cobertura y duplicados
    raw = pd.read_csv(RAW_DIR / "results_latest.csv")
    raw_2000 = (pd.to_datetime(raw["date"], errors="coerce") >= "2000-01-01").sum()
    db_2000 = (matches["date"] >= "2000-01-01").sum()
    dups = matches.duplicated(subset=["date", "home_team_id", "away_team_id"]).sum()
    c1 = db_2000 >= 0.99 * raw_2000 and dups == 0
    ok.append(c1)
    lines += [f"## C1 Cobertura: {'PASS' if c1 else 'FAIL'}",
              f"- Partidos 2000+ en crudo: {raw_2000}; en DB: {db_2000} "
              f"({db_2000 / raw_2000:.2%}); duplicados: {dups}", ""]

    # C2: conciliacion de nombres
    former = set(pd.read_csv(RAW_DIR / "former_names_latest.csv")["former"])
    orphans = sorted(set(teams["name"]) & former)
    c2 = len(orphans) == 0
    ok.append(c2)
    lines += [f"## C2 Conciliacion de nombres: {'PASS' if c2 else 'FAIL'}",
              f"- Nombres 'former' sobrevivientes: {orphans or 'ninguno'}",
              f"- Total equipos: {len(teams)}", ""]

    # C3: Elo vs eloratings.net
    ref = fetch_eloratings()
    if ref is not None:
        joined = ref.merge(elo, on="team")
        corr = joined["elo_ref"].corr(joined["elo"])
        c3 = corr > 0.95
        lines += [f"## C3 Correlacion Elo (n={len(joined)}): "
                  f"{'PASS' if c3 else 'FAIL'} — r = {corr:.4f}", ""]
    else:
        top10 = set(elo.nlargest(10, "elo")["team"])
        hits = len(top10 & ELO_SANITY_SET)
        c3 = hits >= 8
        lines += ["## C3 Correlacion Elo: fuente no disponible — "
                  f"sanity top-10: {'PASS' if c3 else 'FAIL'} ({hits}/10 en set de elite)",
                  f"- Top-10 propio: {sorted(top10)}",
                  "- ACCION: verificar manualmente contra eloratings.net", ""]
    ok.append(c3)

    lines += [f"## Rankings FIFA: {n_rank} filas "
              f"({'cargados' if n_rank else 'NO disponibles — reintentar'})", "",
              f"# GATE F1: {'PASS' if all(ok) else 'FAIL'}"]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = REPORTS_DIR / "f1_validation.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
```

- [ ] **Step 8.2: Corrida real** — `python -m src.etl.validate`
Expected: reporte impreso y guardado; C1 y C2 en PASS; C3 PASS (o sanity PASS con nota de verificación manual).

- [ ] **Step 8.3: Commit** — `git add src/etl/validate.py outputs/reports/f1_validation.md && git commit -m "feat: validacion del gate F1 con reporte"` (forzar el add del reporte si el .gitignore lo excluye: `git add -f outputs/reports/f1_validation.md`).

---

### Task 9: Sedes 2026 (`configs/tournament_2026.yaml`)

**Files:**
- Create: `configs/tournament_2026.yaml`
- Test: `tests/test_tournament_config.py`

- [ ] **Step 9.1: Test que falla** — `tests/test_tournament_config.py`

```python
import yaml

from src.config import CONFIGS_DIR


def test_tournament_2026_has_16_venues_with_coords_and_altitude():
    cfg = yaml.safe_load((CONFIGS_DIR / "tournament_2026.yaml").read_text(encoding="utf-8"))
    venues = cfg["venues"]
    assert len(venues) == 16
    for v in venues:
        assert {"name", "city", "country", "lat", "lon", "altitude_m"} <= set(v)
    countries = {v["country"] for v in venues}
    assert countries == {"United States", "Mexico", "Canada"}
```

- [ ] **Step 9.2:** `pytest tests/test_tournament_config.py -v` → FAIL.

- [ ] **Step 9.3: Crear `configs/tournament_2026.yaml`**

```yaml
# Mundial FIFA 2026 - configuracion del torneo
# Sedes con coordenadas y altitud (m). Grupos y calendario se agregan en Fase 5.
tournament:
  name: FIFA World Cup 2026
  start_date: 2026-06-11
  final_date: 2026-07-19
  n_teams: 48
  hosts: [United States, Mexico, Canada]

venues:
  - {name: Estadio Azteca,          city: Mexico City,     country: Mexico,        lat: 19.3029,  lon: -99.1505,  altitude_m: 2240}
  - {name: Estadio Akron,           city: Guadalajara,     country: Mexico,        lat: 20.6817,  lon: -103.4626, altitude_m: 1560}
  - {name: Estadio BBVA,            city: Monterrey,       country: Mexico,        lat: 25.6695,  lon: -100.2455, altitude_m: 530}
  - {name: BMO Field,               city: Toronto,         country: Canada,        lat: 43.6332,  lon: -79.4186,  altitude_m: 76}
  - {name: BC Place,                city: Vancouver,       country: Canada,        lat: 49.2767,  lon: -123.1119, altitude_m: 9}
  - {name: MetLife Stadium,         city: East Rutherford, country: United States, lat: 40.8135,  lon: -74.0744,  altitude_m: 2}
  - {name: AT&T Stadium,            city: Arlington,       country: United States, lat: 32.7473,  lon: -97.0945,  altitude_m: 168}
  - {name: Arrowhead Stadium,       city: Kansas City,     country: United States, lat: 39.0489,  lon: -94.4839,  altitude_m: 265}
  - {name: NRG Stadium,             city: Houston,         country: United States, lat: 29.6847,  lon: -95.4107,  altitude_m: 15}
  - {name: Mercedes-Benz Stadium,   city: Atlanta,         country: United States, lat: 33.7554,  lon: -84.4010,  altitude_m: 303}
  - {name: Hard Rock Stadium,       city: Miami Gardens,   country: United States, lat: 25.9580,  lon: -80.2389,  altitude_m: 3}
  - {name: Lincoln Financial Field, city: Philadelphia,    country: United States, lat: 39.9008,  lon: -75.1675,  altitude_m: 12}
  - {name: Gillette Stadium,        city: Foxborough,      country: United States, lat: 42.0909,  lon: -71.2643,  altitude_m: 89}
  - {name: Lumen Field,             city: Seattle,         country: United States, lat: 47.5952,  lon: -122.3316, altitude_m: 5}
  - {name: "Levi's Stadium",        city: Santa Clara,     country: United States, lat: 37.4032,  lon: -121.9698, altitude_m: 7}
  - {name: SoFi Stadium,            city: Inglewood,       country: United States, lat: 33.9535,  lon: -118.3392, altitude_m: 35}
```

- [ ] **Step 9.4:** `pytest tests/test_tournament_config.py -v` → PASS.
- [ ] **Step 9.5: Commit** — `git commit -m "feat: sedes del Mundial 2026 con coordenadas y altitud"`

---

### Task 10: Automatización (GitHub Actions) y cierre

**Files:**
- Create: `.github/workflows/update_data.yml`

- [ ] **Step 10.1: Crear workflow**

```yaml
name: Actualizar datos
on:
  schedule:
    - cron: "0 6 * * *"   # diario 06:00 UTC
  workflow_dispatch: {}

jobs:
  etl:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.12"}
      - run: pip install -e .
      - run: python -m src.etl.run_all
      - run: python -m src.etl.validate
      - uses: actions/upload-artifact@v4
        with:
          name: datos-procesados
          path: |
            data/processed/
            outputs/reports/
```

(Sube artefactos en vez de commitear datos: el `.gitignore` excluye `data/`.)

- [ ] **Step 10.2: Suite completa** — `pytest -v` → todos PASS.
- [ ] **Step 10.3: Commit final** — `git add .github/ && git commit -m "ci: actualizacion diaria de datos via GitHub Actions"`
- [ ] **Step 10.4:** Actualizar el checklist del `README.md`: marcar "Fase 1" como completada **solo si** el reporte `f1_validation.md` dice `GATE F1: PASS`. Commit `docs: fase 1 completada (gate PASS)`.

---

## Self-Review (hecho al escribir el plan)

- **Cobertura del spec F1:** repo+estructura (T1), ETL partidos (T2–T4), Elo propio (T5), ranking FIFA (T6), sedes 2026 (T9), GitHub Action (T10), criterios de aceptación (T8). ✔
- **Sin placeholders:** todo código completo. ✔
- **Consistencia de tipos:** `clean_results` produce `date` datetime y `neutral` bool; `load_matches` espera exactamente eso; `compute_elo` usa columnas de `clean_results`. `tidy_rankings` produce las 4 columnas que `run_all` inserta (más `team_id` añadido allí). ✔
- **Nota:** el scraping directo de eloratings.net es solo para validación (C3), con fallback documentado — el Elo del proyecto es propio, sin dependencia externa.
