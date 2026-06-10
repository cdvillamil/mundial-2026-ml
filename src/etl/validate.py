"""Reporte de validacion del gate F1."""
import datetime as dt
import io
import sqlite3

import pandas as pd
import requests

from src.config import DB_PATH, RAW_DIR, REPORTS_DIR

ELO_SANITY_SET = {
    "Argentina", "France", "Spain", "Brazil", "England", "Portugal",
    "Netherlands", "Germany", "Italy", "Belgium", "Uruguay", "Colombia",
    "Croatia", "Morocco", "Japan", "Ecuador", "Mexico", "United States",
}
# Codigos de eloratings.net para selecciones grandes (suficiente para correlacion)
ELORATINGS_CODES = {
    "AR": "Argentina", "FR": "France", "ES": "Spain", "BR": "Brazil",
    "EN": "England", "PT": "Portugal", "NL": "Netherlands", "DE": "Germany",
    "IT": "Italy", "BE": "Belgium", "UY": "Uruguay", "CO": "Colombia",
    "HR": "Croatia", "MA": "Morocco", "JP": "Japan", "MX": "Mexico",
    "US": "United States", "DK": "Denmark", "CH": "Switzerland",
    "AT": "Austria", "EC": "Ecuador", "SN": "Senegal", "TR": "Turkey",
    "NO": "Norway", "SE": "Sweden", "PL": "Poland", "GR": "Greece",
    "RS": "Serbia", "AU": "Australia", "KR": "South Korea",
}


def fetch_eloratings() -> pd.DataFrame | None:
    """World.tsv de eloratings.net: sin header; col 2 = codigo, col 3 = rating."""
    try:
        resp = requests.get("https://www.eloratings.net/World.tsv", timeout=30,
                            headers={"User-Agent": "Mozilla/5.0 (educational project)"})
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text), sep="\t", header=None)
        out = df.iloc[:, [2, 3]].copy()
        out.columns = ["code", "elo_ref"]
        out["elo_ref"] = pd.to_numeric(out["elo_ref"], errors="coerce")
        out["team"] = out["code"].map(ELORATINGS_CODES)
        out = out.dropna(subset=["team", "elo_ref"])
        return out[["team", "elo_ref"]] if len(out) >= 15 else None
    except Exception as exc:
        print(f"WARN: eloratings.net no disponible ({exc})")
        return None


def main() -> None:
    con = sqlite3.connect(DB_PATH)
    matches = pd.read_sql("SELECT * FROM matches", con)
    teams = pd.read_sql("SELECT * FROM teams", con)
    elo = pd.read_sql("""
        SELECT t.name AS team, e.elo FROM elo_history e
        JOIN teams t ON t.team_id = e.team_id
        WHERE e.rowid = (SELECT MAX(x.rowid) FROM elo_history x WHERE x.team_id = e.team_id)
    """, con)
    n_rank = pd.read_sql("SELECT COUNT(*) AS n FROM fifa_rankings", con)["n"][0]
    con.close()

    lines = [f"# Validacion Fase 1 — {dt.date.today().isoformat()}", ""]
    ok = []

    # C1: cobertura y duplicados
    raw = pd.read_csv(RAW_DIR / "results_latest.csv")
    raw_2000 = (pd.to_datetime(raw["date"], errors="coerce") >= "2000-01-01").sum()
    db_2000 = (matches["date"] >= "2000-01-01").sum()
    dups = matches.duplicated(subset=["date", "home_team_id", "away_team_id"]).sum()
    c1 = db_2000 >= 0.99 * raw_2000 and dups == 0
    ok.append(c1)
    lines += [f"## C1 Cobertura: {'PASS' if c1 else 'FAIL'}",
              f"- Partidos 2000+ en crudo: {raw_2000}; en DB: {db_2000} "
              f"({db_2000 / raw_2000:.2%}); duplicados: {dups}", ""]

    # C2: conciliacion de nombres
    former = set(pd.read_csv(RAW_DIR / "former_names_latest.csv")["former"])
    orphans = sorted(set(teams["name"]) & former)
    c2 = len(orphans) == 0
    ok.append(c2)
    lines += [f"## C2 Conciliacion de nombres: {'PASS' if c2 else 'FAIL'}",
              f"- Nombres 'former' sobrevivientes: {orphans or 'ninguno'}",
              f"- Total equipos: {len(teams)}", ""]

    # C3: Elo vs eloratings.net
    ref = fetch_eloratings()
    if ref is not None:
        joined = ref.merge(elo, on="team")
        corr = joined["elo_ref"].corr(joined["elo"])
        c3 = bool(corr > 0.95)
        lines += [f"## C3 Correlacion Elo (n={len(joined)}): "
                  f"{'PASS' if c3 else 'FAIL'} — r = {corr:.4f}", ""]
    else:
        top10 = set(elo.nlargest(10, "elo")["team"])
        hits = len(top10 & ELO_SANITY_SET)
        c3 = hits >= 8
        lines += ["## C3 Correlacion Elo: fuente no disponible — "
                  f"sanity top-10: {'PASS' if c3 else 'FAIL'} ({hits}/10 en set de elite)",
                  f"- Top-10 propio: {sorted(top10)}",
                  "- ACCION: verificar manualmente contra eloratings.net", ""]
    ok.append(c3)

    lines += [f"## Rankings FIFA: {n_rank} filas "
              f"({'cargados' if n_rank else 'NO disponibles — reintentar'})", "",
              f"# GATE F1: {'PASS' if all(ok) else 'FAIL'}"]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "f1_validation.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
