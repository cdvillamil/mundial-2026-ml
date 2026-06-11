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


def _latest_elo_before(elo_sorted: pd.DataFrame, team: str, date) -> float:
    """Ultimo Elo del equipo estrictamente antes de date (point-in-time)."""
    sub = elo_sorted[(elo_sorted["team"] == team) & (elo_sorted["date"] < date)]
    if len(sub) == 0:
        return DEFAULT_ELO
    return float(sub["elo"].iloc[-1])


def attach_pre_match_elo(matches: pd.DataFrame, elo_history: pd.DataFrame) -> pd.DataFrame:
    """Agrega home_elo, away_elo, elo_diff, tournament_importance a matches."""
    elo_sorted = elo_history.sort_values(["team", "date"]).reset_index(drop=True)
    out = matches.copy().reset_index(drop=True)

    home_elos, away_elos = [], []
    for m in out.itertuples(index=False):
        home_elos.append(_latest_elo_before(elo_sorted, m.home_team, m.date))
        away_elos.append(_latest_elo_before(elo_sorted, m.away_team, m.date))

    out["home_elo"] = home_elos
    out["away_elo"] = away_elos
    out["elo_diff"] = out["home_elo"] - out["away_elo"]
    out["tournament_importance"] = out["tournament"].map(tournament_importance)
    return out
