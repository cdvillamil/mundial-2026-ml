"""Carga el sorteo OFICIAL del Mundial 2026 en configs/groups_2026.yaml.

Grupos A-L confirmados tras el sorteo (dic 2025) y repechajes (mar 2026).
Elo tomado de la base de datos (ultimo disponible por equipo).
"""
import sqlite3

import pandas as pd
import yaml

from src.config import CONFIGS_DIR, DB_PATH

# Grupos oficiales (nombre canonico martj42)
OFFICIAL_GROUPS = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}


def latest_elos(con) -> dict:
    df = pd.read_sql(
        """SELECT t.name AS team, e.elo FROM elo_history e
           JOIN teams t ON t.team_id = e.team_id
           WHERE e.rowid = (SELECT MAX(x.rowid) FROM elo_history x
                            WHERE x.team_id = e.team_id)""", con)
    return dict(zip(df["team"], df["elo"]))


def main():
    con = sqlite3.connect(DB_PATH)
    elos_all = latest_elos(con)
    con.close()

    teams = [t for g in OFFICIAL_GROUPS.values() for t in g]
    missing = [t for t in teams if t not in elos_all]
    elos = {t: round(float(elos_all.get(t, 1500.0)), 1) for t in teams}

    field = {
        "note": "Sorteo OFICIAL Mundial 2026 (grupos A-L). Elo desde la base de datos.",
        "official": True,
        "elos": elos,
        "groups": OFFICIAL_GROUPS,
    }
    out = CONFIGS_DIR / "groups_2026.yaml"
    out.write_text(yaml.safe_dump(field, allow_unicode=True, sort_keys=False),
                   encoding="utf-8")
    print(f"Sorteo oficial escrito -> {out}")
    if missing:
        print(f"ADVERTENCIA: sin Elo (default 1500): {missing}")
    for g, ts in OFFICIAL_GROUPS.items():
        print(f"  Grupo {g}: {[(t, elos[t]) for t in ts]}")


if __name__ == "__main__":
    main()
