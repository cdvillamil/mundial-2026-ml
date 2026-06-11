# Fase 7 — Predicción 2026, Dashboard y Evaluación Final: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Generar las predicciones del Mundial 2026 (marcador exacto + 1X2 por partido de grupos, pre-registradas con timestamp), un dashboard visual (Plotly + Streamlit) y el marco de evaluación predicho-vs-real. Gate F7: predicciones 2026 generadas y guardadas; dashboard/reporte HTML funcional; marco de evaluación testeado; suite completa en verde.

**Architecture:**
- `src/inference/predict_2026.py` — para cada partido de grupos del campo 2026, predice matriz de marcadores, marcador más probable y P(1X2); guarda CSV en `outputs/predictions/` con timestamp (pre-registro). Reusa GBM + RateProvider.
- `src/evaluation/evaluate_predictions.py` — compara predicciones pre-registradas vs resultados reales (cuando existan en `data/external/wc2026_results.csv`); calcula RPS, log-loss, accuracy 1X2 y acierto de marcador exacto. Maneja "aún sin resultados".
- `src/dashboard/figures.py` — figuras Plotly puras (barra de P(campeón), heatmap de probabilidades por fase, heatmap de matriz de marcador). Testeable.
- `src/dashboard/data.py` — carga simulación (f5), predicciones 2026 y campo. Testeable.
- `src/dashboard/report.py` — `build_html_report()` escribe `outputs/reports/f7_dashboard.html` (interactivo, sin servidor). **Entregable garantizado.**
- `src/dashboard/app.py` — app Streamlit usando data+figures (corre si Streamlit está instalado).

**Decisión documentada:** el núcleo visual es Plotly (figuras + reporte HTML autónomo), robusto y sin servidor. La app Streamlit es una capa encima; si Streamlit no instala en Python 3.14, el reporte HTML es el entregable equivalente (mismas figuras).

**Tech Stack:** Plotly (siempre), Streamlit (intento). Reusa `GBMPoissonModel`, `RateProvider`, `simulate_tournament`, `score_matrix`, métricas.

---

### Task 1: Dependencias (plotly; intento streamlit)

- [ ] **Step 1.1:** Añadir `"plotly>=5.20"` a `dependencies` en `pyproject.toml`.
- [ ] **Step 1.2:** `.venv\Scripts\python -m pip install -e ".[dev]"` → instala plotly.
- [ ] **Step 1.3:** Intentar `pip install streamlit`. Si instala, añadirlo a deps; si falla en Python 3.14, documentar en el reporte que el entregable visual es el HTML de Plotly (no bloquea la fase).
- [ ] **Step 1.4: Commit** — `git commit -m "build: anadir plotly (y streamlit si disponible)"`

---

### Task 2: Predicciones 2026 (`src/inference/predict_2026.py`)

**Files:** Create `src/inference/__init__.py`, `src/inference/predict_2026.py`, `tests/test_predict_2026.py`

- [ ] **Step 2.1: Test que falla** — `tests/test_predict_2026.py`

```python
import numpy as np
import pandas as pd

from src.models.gbm_poisson import GBMPoissonModel
from src.inference.predict_2026 import predict_group_matches
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


def test_predict_group_matches_columns_and_probs():
    groups = {"A": ["X", "Y", "Z"]}
    elos = {"X": 1900, "Y": 1700, "Z": 1500}
    rp = RateProvider(_model(), elos)
    df = predict_group_matches(groups, rp)
    # 3 partidos en un grupo de 3 (round robin)
    assert len(df) == 3
    assert {"group", "home", "away", "p_home", "p_draw", "p_away",
            "most_likely", "score_prob"} <= set(df.columns)
    # probabilidades suman ~1
    assert np.allclose(df["p_home"] + df["p_draw"] + df["p_away"], 1.0, atol=1e-6)
    # formato de marcador "i-j"
    assert df["most_likely"].str.match(r"^\d+-\d+$").all()


def test_stronger_home_has_higher_p_home():
    groups = {"A": ["X", "Z"]}
    rp = RateProvider(_model(), {"X": 2000, "Z": 1400})
    df = predict_group_matches(groups, rp)
    row = df.iloc[0]
    assert row["p_home"] > row["p_away"]
```

- [ ] **Step 2.2:** `pytest tests/test_predict_2026.py -v` → FAIL

- [ ] **Step 2.3: Implementación** — `src/inference/predict_2026.py`

```python
"""Predicciones del Mundial 2026: marcador exacto + 1X2 por partido de grupos."""
import datetime as dt
import itertools

import pandas as pd

from src.models.score_matrix import most_likely_score, outcome_probs
from src.simulation.rates import RateProvider


def predict_group_matches(groups: dict, rp: RateProvider) -> pd.DataFrame:
    """Para cada emparejamiento de grupo, predice 1X2 y marcador mas probable."""
    rows = []
    for g, teams in groups.items():
        for home, away in itertools.combinations(teams, 2):
            matrix = rp.matrix(home, away)
            p_home, p_draw, p_away = outcome_probs(matrix)
            i, j = most_likely_score(matrix)
            rows.append({
                "group": g, "home": home, "away": away,
                "p_home": round(p_home, 4), "p_draw": round(p_draw, 4),
                "p_away": round(p_away, 4),
                "most_likely": f"{i}-{j}",
                "score_prob": round(float(matrix[i, j]), 4),
            })
    return pd.DataFrame(rows)


def main():
    from src.config import CONFIGS_DIR, PROJECT_ROOT
    from src.simulation.monte_carlo import _load_field_and_model

    field, model = _load_field_and_model()
    rp = RateProvider(model, {t: float(e) for t, e in field["elos"].items()})
    df = predict_group_matches(field["groups"], rp)

    out_dir = PROJECT_ROOT / "outputs" / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    path = out_dir / f"group_stage_2026_{stamp}.csv"
    df.to_csv(path, index=False, encoding="utf-8")
    # copia estable "latest" para el dashboard
    df.to_csv(out_dir / "group_stage_2026_latest.csv", index=False, encoding="utf-8")
    print(f"Pre-registradas {len(df)} predicciones de grupos -> {path}")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2.4:** `pytest tests/test_predict_2026.py -v` → PASS
- [ ] **Step 2.5: Corrida real** — `python -m src.inference.predict_2026` → 72 predicciones (12 grupos × 6).
- [ ] **Step 2.6: Commit** — `git add src/inference/ tests/test_predict_2026.py outputs/predictions/group_stage_2026_latest.csv` (con -f) `&& git commit -m "feat: predicciones pre-registradas de la fase de grupos 2026"`

---

### Task 3: Marco de evaluación predicho-vs-real (`src/evaluation/evaluate_predictions.py`)

**Files:** Create `src/evaluation/evaluate_predictions.py`, `tests/test_evaluate_predictions.py`

- [ ] **Step 3.1: Test que falla** — `tests/test_evaluate_predictions.py`

```python
import pandas as pd

from src.evaluation.evaluate_predictions import evaluate


def _preds():
    return pd.DataFrame({
        "home": ["A", "C"], "away": ["B", "D"],
        "p_home": [0.7, 0.2], "p_draw": [0.2, 0.3], "p_away": [0.1, 0.5],
        "most_likely": ["2-0", "0-1"],
    })


def test_evaluate_with_results():
    results = pd.DataFrame({
        "home": ["A", "C"], "away": ["B", "D"],
        "home_goals": [2, 1], "away_goals": [0, 1],  # A gana 2-0 (exacto), C-D empate
    })
    rep = evaluate(_preds(), results)
    assert rep["n_evaluated"] == 2
    assert rep["exact_score_acc"] == 0.5   # solo A 2-0 exacto
    assert 0 <= rep["rps"] <= 1
    assert 0 <= rep["accuracy_1x2"] <= 1


def test_evaluate_no_results_yet():
    empty = pd.DataFrame(columns=["home", "away", "home_goals", "away_goals"])
    rep = evaluate(_preds(), empty)
    assert rep["n_evaluated"] == 0
    assert rep["status"] == "sin resultados aun"
```

- [ ] **Step 3.2:** `pytest tests/test_evaluate_predictions.py -v` → FAIL

- [ ] **Step 3.3: Implementación** — `src/evaluation/evaluate_predictions.py`

```python
"""Compara predicciones pre-registradas vs resultados reales del Mundial 2026."""
import numpy as np
import pandas as pd

from src.evaluation.metrics import accuracy_1x2, log_loss_1x2, rps


def _outcome(hg, ag):
    return 0 if hg > ag else (1 if hg == ag else 2)


def evaluate(preds: pd.DataFrame, results: pd.DataFrame) -> dict:
    """Une predicciones con resultados reales (por home/away) y calcula metricas."""
    if len(results) == 0:
        return {"n_evaluated": 0, "status": "sin resultados aun"}

    merged = preds.merge(results, on=["home", "away"], how="inner")
    if len(merged) == 0:
        return {"n_evaluated": 0, "status": "sin coincidencias"}

    probs = merged[["p_home", "p_draw", "p_away"]].to_numpy()
    actual = np.array([_outcome(h, a) for h, a in
                       zip(merged["home_goals"], merged["away_goals"])])
    exact = (merged["most_likely"]
             == merged["home_goals"].astype(str) + "-" + merged["away_goals"].astype(str))
    return {
        "n_evaluated": int(len(merged)),
        "status": "evaluado",
        "rps": round(rps(probs, actual), 4),
        "log_loss": round(log_loss_1x2(probs, actual), 4),
        "accuracy_1x2": round(accuracy_1x2(probs, actual), 4),
        "exact_score_acc": round(float(exact.mean()), 4),
    }


def main():
    from src.config import PROJECT_ROOT
    preds = pd.read_csv(PROJECT_ROOT / "outputs" / "predictions"
                        / "group_stage_2026_latest.csv")
    results_path = PROJECT_ROOT / "data" / "external" / "wc2026_results.csv"
    if results_path.exists():
        results = pd.read_csv(results_path)
    else:
        results = pd.DataFrame(columns=["home", "away", "home_goals", "away_goals"])
    rep = evaluate(preds, results)
    print(rep)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3.4:** `pytest tests/test_evaluate_predictions.py -v` → PASS
- [ ] **Step 3.5: Commit** — `git commit -m "feat: marco de evaluacion predicho-vs-real (RPS, marcador exacto)"`

---

### Task 4: Figuras Plotly (`src/dashboard/figures.py`)

**Files:** Create `src/dashboard/__init__.py`, `src/dashboard/figures.py`, `tests/test_figures.py`

- [ ] **Step 4.1: Test que falla** — `tests/test_figures.py`

```python
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.dashboard.figures import champion_bar, score_matrix_heatmap


def test_champion_bar_returns_figure():
    df = pd.DataFrame({"team": ["A", "B", "C"], "p_champion": [0.3, 0.2, 0.1]})
    fig = champion_bar(df, top=3)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1


def test_score_matrix_heatmap_returns_figure():
    m = np.full((6, 6), 1 / 36)
    fig = score_matrix_heatmap(m, "A", "B")
    assert isinstance(fig, go.Figure)
```

- [ ] **Step 4.2:** `pytest tests/test_figures.py -v` → FAIL

- [ ] **Step 4.3: Implementación** — `src/dashboard/figures.py`

```python
"""Figuras Plotly del dashboard (puras, sin estado/servidor)."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def champion_bar(sim_results: pd.DataFrame, top: int = 12) -> go.Figure:
    """Barra horizontal de P(campeon) para los 'top' equipos."""
    d = sim_results.nlargest(top, "p_champion").iloc[::-1]
    fig = go.Figure(go.Bar(
        x=(d["p_champion"] * 100), y=d["team"], orientation="h",
        text=[f"{v*100:.1f}%" for v in d["p_champion"]], textposition="auto"))
    fig.update_layout(title="Probabilidad de ser campeon (%)",
                      xaxis_title="%", yaxis_title="", height=500)
    return fig


def phase_heatmap(sim_results: pd.DataFrame, top: int = 16) -> go.Figure:
    """Heatmap de probabilidades por fase para los 'top' equipos."""
    cols = ["p_qualify", "p_r16", "p_qf", "p_sf", "p_final", "p_champion"]
    labels = ["Clasifica", "Octavos", "Cuartos", "Semis", "Final", "Campeon"]
    d = sim_results.nlargest(top, "p_champion")
    z = (d[cols].to_numpy() * 100)
    fig = go.Figure(go.Heatmap(
        z=z, x=labels, y=d["team"].tolist(),
        colorscale="Blues", text=np.round(z, 1), texttemplate="%{text}"))
    fig.update_layout(title="Probabilidad de avance por fase (%)",
                      height=600, yaxis=dict(autorange="reversed"))
    return fig


def score_matrix_heatmap(matrix: np.ndarray, home: str, away: str,
                         max_goals: int = 5) -> go.Figure:
    """Heatmap de la matriz de probabilidades de marcador (recortada)."""
    m = matrix[:max_goals + 1, :max_goals + 1]
    z = m * 100
    fig = go.Figure(go.Heatmap(
        z=z, x=[str(j) for j in range(max_goals + 1)],
        y=[str(i) for i in range(max_goals + 1)],
        colorscale="YlOrRd", text=np.round(z, 1), texttemplate="%{text}"))
    fig.update_layout(title=f"Prob. de marcador: {home} (filas) vs {away} (col)",
                      xaxis_title=f"Goles {away}", yaxis_title=f"Goles {home}",
                      height=450)
    return fig
```

- [ ] **Step 4.4:** `pytest tests/test_figures.py -v` → PASS
- [ ] **Step 4.5: Commit** — `git commit -m "feat: figuras Plotly del dashboard (campeon, fases, marcador)"`

---

### Task 5: Datos, reporte HTML y app Streamlit

**Files:** Create `src/dashboard/data.py`, `src/dashboard/report.py`, `src/dashboard/app.py`

- [ ] **Step 5.1: Implementación** — `src/dashboard/data.py`

```python
"""Carga de datos para el dashboard."""
import pandas as pd
import yaml

from src.config import CONFIGS_DIR, PROJECT_ROOT


def load_field() -> dict:
    return yaml.safe_load((CONFIGS_DIR / "groups_2026.yaml").read_text(encoding="utf-8"))


def load_predictions() -> pd.DataFrame:
    path = PROJECT_ROOT / "outputs" / "predictions" / "group_stage_2026_latest.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def build_rate_provider():
    """Entrena el modelo y construye el RateProvider para 2026 (matrices on-demand)."""
    from src.simulation.monte_carlo import _load_field_and_model
    from src.simulation.rates import RateProvider
    field, model = _load_field_and_model()
    return field, RateProvider(model, {t: float(e) for t, e in field["elos"].items()})


def run_simulation(field, rp, n_sims=50000):
    from src.simulation.monte_carlo import simulate_tournament
    return simulate_tournament(field["groups"], rp, n_sims=n_sims,
                               n_qualify_per_group=2, n_best_thirds=8, seed=42)
```

- [ ] **Step 5.2: Implementación** — `src/dashboard/report.py`

```python
"""Genera un reporte HTML interactivo autonomo (sin servidor)."""
import pandas as pd

from src.config import PROJECT_ROOT
from src.dashboard.figures import champion_bar, phase_heatmap


def build_html_report(sim_results: pd.DataFrame, preds: pd.DataFrame) -> str:
    out = PROJECT_ROOT / "outputs" / "reports" / "f7_dashboard.html"
    out.parent.mkdir(parents=True, exist_ok=True)

    parts = ["<html><head><meta charset='utf-8'>",
             "<title>Mundial 2026 - Predicciones</title></head><body>",
             "<h1>Prediccion del Mundial FIFA 2026</h1>",
             "<p><b>Campo de ejemplo sembrado por Elo (no es el sorteo oficial).</b></p>",
             champion_bar(sim_results).to_html(full_html=False, include_plotlyjs="cdn"),
             phase_heatmap(sim_results).to_html(full_html=False, include_plotlyjs=False),
             "<h2>Predicciones de la fase de grupos (marcador mas probable)</h2>",
             preds.to_html(index=False)]
    parts.append("</body></html>")
    html = "\n".join(parts)
    out.write_text(html, encoding="utf-8")
    return str(out)


def main():
    from src.dashboard.data import build_rate_provider, load_predictions, run_simulation
    field, rp = build_rate_provider()
    sim = run_simulation(field, rp, n_sims=50000)
    preds = load_predictions()
    path = build_html_report(sim, preds)
    print(f"Reporte HTML generado: {path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5.3: Implementación** — `src/dashboard/app.py`

```python
"""Dashboard Streamlit del Mundial 2026 (corre si streamlit esta instalado:
   streamlit run src/dashboard/app.py)."""
import streamlit as st

from src.dashboard.data import build_rate_provider, load_predictions, run_simulation
from src.dashboard.figures import champion_bar, phase_heatmap, score_matrix_heatmap


@st.cache_resource
def _setup():
    field, rp = build_rate_provider()
    sim = run_simulation(field, rp, n_sims=50000)
    return field, rp, sim


def main():
    st.set_page_config(page_title="Mundial 2026 ML", layout="wide")
    st.title("Prediccion del Mundial FIFA 2026")
    st.caption("Campo de ejemplo sembrado por Elo (no es el sorteo oficial).")

    field, rp, sim = _setup()

    st.header("Probabilidad de ser campeon")
    st.plotly_chart(champion_bar(sim), use_container_width=True)

    st.header("Probabilidad de avance por fase")
    st.plotly_chart(phase_heatmap(sim), use_container_width=True)

    st.header("Marcador mas probable de un partido")
    teams = sorted(field["elos"])
    c1, c2 = st.columns(2)
    home = c1.selectbox("Local", teams, index=0)
    away = c2.selectbox("Visitante", teams, index=1)
    if home != away:
        st.plotly_chart(score_matrix_heatmap(rp.matrix(home, away), home, away),
                        use_container_width=True)

    st.header("Predicciones de la fase de grupos")
    st.dataframe(load_predictions(), use_container_width=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5.4: Corrida real** — `python -m src.dashboard.report` → genera `outputs/reports/f7_dashboard.html`. Verificar que el archivo existe y pesa > 50 KB.
- [ ] **Step 5.5: Commit** — `git add src/dashboard/ && git add -f outputs/reports/f7_dashboard.html && git commit -m "feat: dashboard (reporte HTML Plotly + app Streamlit)"`

---

### Task 6: Validación y cierre F7 (proyecto completo)

- [ ] **Step 6.1:** `pytest -v` → todos PASS
- [ ] **Step 6.2:** Verificar entregables F7: predicciones 2026 en `outputs/predictions/`, reporte HTML en `outputs/reports/f7_dashboard.html`, app Streamlit importable, evaluador testeado.
- [ ] **Step 6.3:** Crear plantilla `data/external/wc2026_results.csv` (cabecera `home,away,home_goals,away_goals`) para que el usuario llene resultados conforme avance el torneo, y documentar el flujo de evaluación en README.
- [ ] **Step 6.4:** Actualizar README: marcar Fase 7, añadir sección "Cómo ver las predicciones" (HTML + streamlit) y "Cómo evaluar contra resultados reales".
- [ ] **Step 6.5: Commit + push** — `git commit -m "docs: fase 7 completada — proyecto end-to-end"; git push`

---

## Self-Review

- **Cobertura del spec F7 (§8):** predicciones 2026 pre-registradas con timestamp ✅, dashboard (probabilidades por equipo/fase + matriz de marcador por partido) ✅, comparación predicho-vs-real ✅, marco de evaluación ✅.
- **Reuso:** GBM + RateProvider + simulate_tournament (F5), score_matrix (F3), métricas (F3) — sin reimplementar.
- **Decisión robusta:** Plotly + reporte HTML como entregable garantizado; Streamlit como capa opcional (riesgo de instalación en Python 3.14 mitigado).
- **Pre-registro honesto:** las predicciones se guardan con timestamp; el evaluador maneja "sin resultados aún" (el torneo no está en la base de datos).
- **Cierre del proyecto:** plantilla de resultados + flujo documentado para evaluar la precisión real del Mundial 2026 conforme se juegue.
- **Sin placeholders:** código completo en cada step.
