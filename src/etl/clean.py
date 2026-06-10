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
