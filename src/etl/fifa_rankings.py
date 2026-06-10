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
