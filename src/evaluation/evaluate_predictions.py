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
