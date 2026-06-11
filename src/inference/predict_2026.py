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
    from src.config import PROJECT_ROOT
    from src.simulation.monte_carlo import _load_field_and_model

    field, model = _load_field_and_model()
    rp = RateProvider(model, {t: float(e) for t, e in field["elos"].items()})
    df = predict_group_matches(field["groups"], rp)

    out_dir = PROJECT_ROOT / "outputs" / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    path = out_dir / f"group_stage_2026_{stamp}.csv"
    df.to_csv(path, index=False, encoding="utf-8")
    df.to_csv(out_dir / "group_stage_2026_latest.csv", index=False, encoding="utf-8")
    print(f"Pre-registradas {len(df)} predicciones de grupos -> {path}")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
