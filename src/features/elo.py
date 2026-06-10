"""Elo propio segun la formula de eloratings.net (World Football Elo)."""
import pandas as pd

INITIAL_ELO = 1500.0
HOME_ADVANTAGE = 100.0

_CONTINENTAL_FINALS = {
    "UEFA Euro", "Copa América", "African Cup of Nations", "AFC Asian Cup",
    "CONCACAF Championship", "Gold Cup", "Oceania Nations Cup",
    "Confederations Cup",
}


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
