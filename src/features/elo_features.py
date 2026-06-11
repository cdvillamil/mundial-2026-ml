"""Features de Elo point-in-time (anti-leakage) para cada partido."""
import pandas as pd

DEFAULT_ELO = 1500.0

_CONTINENTAL = {
    "UEFA Euro", "Copa América", "African Cup of Nations", "AFC Asian Cup",
    "CONCACAF Championship", "Gold Cup", "Oceania Nations Cup", "Confederations Cup",
}


def tournament_importance(tournament: str) -> int:
    """0=amistoso, 1=eliminatoria, 2=copa continental, 3=mundial."""
    if tournament == "FIFA World Cup":
        return 3
    if tournament in _CONTINENTAL:
        return 2
    if "qualification" in tournament.lower():
        return 1
    if tournament == "Friendly":
        return 0
    return 1


def _merge_side_elo(matches: pd.DataFrame, elo_sorted: pd.DataFrame,
                    team_col: str, out_col: str) -> pd.Series:
    """Elo point-in-time (estrictamente anterior) via merge_asof, O(n log n)."""
    left = (matches[["date", team_col]]
            .rename(columns={team_col: "team"})
            .reset_index()  # preserva el indice original como columna 'index'
            .sort_values("date"))
    merged = pd.merge_asof(
        left, elo_sorted, on="date", by="team",
        direction="backward", allow_exact_matches=False,  # estricto: elo < fecha partido
    )
    return (merged.set_index("index")["elo"]
            .reindex(matches.index).fillna(DEFAULT_ELO).rename(out_col))


def attach_pre_match_elo(matches: pd.DataFrame, elo_history: pd.DataFrame) -> pd.DataFrame:
    """Agrega home_elo, away_elo, elo_diff, tournament_importance a matches."""
    elo_sorted = elo_history.sort_values(["date", "team"]).reset_index(drop=True)
    out = matches.copy()

    out["home_elo"] = _merge_side_elo(out, elo_sorted, "home_team", "home_elo").to_numpy()
    out["away_elo"] = _merge_side_elo(out, elo_sorted, "away_team", "away_elo").to_numpy()
    out["elo_diff"] = out["home_elo"] - out["away_elo"]
    out["tournament_importance"] = out["tournament"].map(tournament_importance)
    return out
