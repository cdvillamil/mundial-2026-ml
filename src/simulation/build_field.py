"""Genera un campo de 48 equipos sembrado en 12 grupos desde el Elo del DB.

CAMPO DE EJEMPLO: anfitriones (USA, Mexico, Canada) + top 45 por Elo.
Reemplazar 'groups' en configs/groups_2026.yaml con el sorteo oficial cuando se tenga.
"""
import sqlite3

import pandas as pd
import yaml

from src.config import CONFIGS_DIR, DB_PATH

HOSTS = ["United States", "Mexico", "Canada"]
GROUP_LETTERS = list("ABCDEFGHIJKL")  # 12 grupos

# Entidades no-FIFA o desaparecidas que aparecen en el historial de partidos
# (juegan amistosos o son selecciones historicas) — excluir del campo 2026.
NON_FIFA = {
    "Basque Country", "Catalonia", "Galicia", "Jersey", "Guernsey",
    "Isle of Man", "Greenland", "Zanzibar", "Yugoslavia", "Czechoslovakia",
    "Soviet Union", "East Germany", "Saarland", "Northern Cyprus", "Kosovo",
    "Tibet", "Monaco", "Kiribati", "Tuvalu", "Niue", "Sealand",
}
ACTIVE_SINCE = "2023-01-01"  # solo selecciones activas recientemente


def latest_elos() -> pd.DataFrame:
    """Elo mas reciente por equipo + fecha de su ultimo partido (para recencia)."""
    con = sqlite3.connect(DB_PATH)
    elo = pd.read_sql(
        """SELECT t.name AS team, e.elo, e.date AS last_date FROM elo_history e
           JOIN teams t ON t.team_id = e.team_id
           WHERE e.rowid = (SELECT MAX(x.rowid) FROM elo_history x
                            WHERE x.team_id = e.team_id)""", con)
    con.close()
    return elo


def build_field() -> dict:
    elo_df = latest_elos()
    # filtrar: activos desde ACTIVE_SINCE y miembros FIFA elegibles
    elo_df = elo_df[(elo_df["last_date"] >= ACTIVE_SINCE)
                    & (~elo_df["team"].isin(NON_FIFA))]
    elo = elo_df.set_index("team")["elo"].to_dict()
    # anfitriones primero (clasifican de oficio), luego top por Elo hasta completar 48
    field = list(HOSTS)
    for team, _ in sorted(elo.items(), key=lambda kv: -kv[1]):
        if team not in field:
            field.append(team)
        if len(field) == 48:
            break

    # siembra serpiente por Elo en 12 grupos (4 por grupo)
    field_sorted = sorted(field, key=lambda t: -elo.get(t, 1500))
    groups = {g: [] for g in GROUP_LETTERS}
    for i, team in enumerate(field_sorted):
        row = i // 12
        pos = i % 12
        g = GROUP_LETTERS[pos] if row % 2 == 0 else GROUP_LETTERS[11 - pos]
        groups[g].append(team)

    return {
        "note": "CAMPO DE EJEMPLO sembrado por Elo. Reemplazar con el sorteo oficial 2026.",
        "elos": {t: round(elo.get(t, 1500), 1) for t in field},
        "groups": groups,
    }


def main():
    field = build_field()
    out = CONFIGS_DIR / "groups_2026.yaml"
    out.write_text(yaml.safe_dump(field, allow_unicode=True, sort_keys=False),
                   encoding="utf-8")
    print(f"Campo generado: {out}")
    for g, teams in field["groups"].items():
        print(f"  Grupo {g}: {teams}")


if __name__ == "__main__":
    main()
