"""Calculo de forma reciente (victorias, derrotas, goles) en ventanas moviles."""
import pandas as pd


def compute_form(matches: pd.DataFrame, team_col: str, windows: list[int] = [5, 10, 20]
                 ) -> pd.DataFrame:
    """
    Para cada fila (partido), calcula estadísticas de los últimos N partidos
    del equipo especificado (antes de esa fecha, sin leakage).

    Retorna dataframe con columnas form_5_wins, form_5_draws, form_5_losses,
    form_5_gf, form_5_ga, etc.
    """
    matches = matches.sort_values("date").reset_index(drop=True)
    rows = []

    for idx, match in matches.iterrows():
        team = match[team_col]
        match_date = match["date"]

        # Partidos anteriores de este equipo
        prev = matches[
            (matches["date"] < match_date) &
            ((matches["home_team"] == team) | (matches["away_team"] == team))
        ].tail(max(windows))

        row = {"match_idx": idx}
        for w in windows:
            recent = prev.tail(w)
            if len(recent) < w:
                row.update({
                    f"form_{w}_wins": None, f"form_{w}_draws": None,
                    f"form_{w}_losses": None, f"form_{w}_gf": None,
                    f"form_{w}_ga": None,
                })
                continue

            wins = sum(
                ((recent["home_team"] == team) & (recent["home_goals"] > recent["away_goals"])) |
                ((recent["away_team"] == team) & (recent["away_goals"] > recent["home_goals"]))
            )
            draws = sum(recent["home_goals"] == recent["away_goals"])
            losses = len(recent) - wins - draws
            gf = recent[recent["home_team"] == team]["home_goals"].sum() + \
                 recent[recent["away_team"] == team]["away_goals"].sum()
            ga = recent[recent["home_team"] == team]["away_goals"].sum() + \
                 recent[recent["away_team"] == team]["home_goals"].sum()

            row.update({
                f"form_{w}_wins": wins, f"form_{w}_draws": draws,
                f"form_{w}_losses": losses, f"form_{w}_gf": gf, f"form_{w}_ga": ga,
            })
        rows.append(row)

    return pd.DataFrame(rows)
