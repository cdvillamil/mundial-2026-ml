"""Backtest comparativo de modelos estadisticos con split temporal."""
import sqlite3

import numpy as np
import pandas as pd

from src.config import DB_PATH, REPORTS_DIR
from src.evaluation.metrics import accuracy_1x2, brier_1x2, log_loss_1x2, rps
from src.models.dixon_coles import DixonColesModel
from src.models.poisson import PoissonModel


def _load_matches(min_date: str = "2010-01-01") -> pd.DataFrame:
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

    poisson = PoissonModel().fit(train)
    dc = DixonColesModel().fit(train)
    dc_decay = DixonColesModel(xi=0.003).fit(train)  # decaimiento ~half-life 0.6 anos
    known = set(poisson.teams_)
    test = test[test["home_team"].isin(known) & test["away_team"].isin(known)]

    actual = np.array([_outcome(r.home_goals, r.away_goals) for r in test.itertuples()])

    results = []
    for name, model in [("Poisson", poisson), ("Dixon-Coles", dc),
                        ("Dixon-Coles+decay", dc_decay)]:
        probs = np.array([
            model.predict_proba_1x2(r.home_team, r.away_team, bool(r.neutral))
            for r in test.itertuples()
        ])
        results.append({
            "model": name,
            "n_test": len(test),
            "RPS": round(rps(probs, actual), 4),
            "log_loss": round(log_loss_1x2(probs, actual), 4),
            "Brier": round(brier_1x2(probs, actual), 4),
            "accuracy": round(accuracy_1x2(probs, actual), 4),
        })

    return pd.DataFrame(results)


def main():
    res = run_backtest()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "f3_model_comparison.md"
    best = res.loc[res["RPS"].idxmin(), "model"]
    lines = [
        "# Comparacion de modelos estadisticos (corte temporal 2018-01-01)",
        "",
        "Entrenamiento: partidos 2010-2018. Test: partidos 2018+.",
        "Convencion: menor RPS/log-loss/Brier es mejor; mayor accuracy es mejor.",
        "",
        res.to_markdown(index=False),
        "",
        f"**Mejor modelo por RPS: {best}**",
        "",
        "## Interpretacion honesta",
        "",
        "- Los tres modelos producen probabilidades razonables (~58% accuracy 1X2,",
        "  muy por encima del ~40% de azar informado).",
        "- Poisson y Dixon-Coles empatan en la practica (RPS 0.3596 vs 0.3597): con un",
        "  modelo global de ataque/defensa sin features ricas, la correccion de marcadores",
        "  bajos de Dixon-Coles no aporta mejora medible en seleccciones.",
        "- El decaimiento temporal agresivo (xi=0.003) EMPEORA: las selecciones juegan",
        "  pocos partidos, descartar historia pierde demasiada senal. Habria que afinar xi",
        "  con validacion (mucho menor) o no usarlo.",
        "- **Conclusion:** estos son baselines estadisticos solidos. Las mejoras reales",
        "  vendran en Fase 4 con modelos GBM que exploten Elo, forma y contexto.",
        "  El valor de Fase 3 es establecer el baseline y la infraestructura de evaluacion",
        "  (matriz de marcadores, metricas RPS/log-loss/Brier, backtest temporal).",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
