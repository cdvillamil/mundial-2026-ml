"""Pipeline ETL completo: descarga -> limpieza -> carga -> Elo -> rankings."""
import sqlite3

import pandas as pd

from src.config import DB_PATH, PROCESSED_DIR, RAW_DIR
from src.etl.clean import build_alias_map, clean_results, clean_shootouts
from src.etl.download import download_all
from src.etl.fifa_rankings import download_rankings, tidy_rankings
from src.etl.load import load_matches
from src.features.elo import compute_elo

# Nombres del dataset FIFA -> nombres canonicos martj42
FIFA_ALIASES = {
    "IR Iran": "Iran", "Korea Republic": "South Korea",
    "Korea DPR": "North Korea", "USA": "United States",
    "Côte d'Ivoire": "Ivory Coast", "Cabo Verde": "Cape Verde",
    "China PR": "China", "Congo DR": "DR Congo",
    "Türkiye": "Turkey", "Czechia": "Czech Republic",
    "St. Kitts and Nevis": "Saint Kitts and Nevis",
    "St. Lucia": "Saint Lucia",
    "St. Vincent / Grenadines": "Saint Vincent and the Grenadines",
    "Brunei Darussalam": "Brunei", "Kyrgyz Republic": "Kyrgyzstan",
    "Hong Kong, China": "Hong Kong", "Macau, China": "Macau",
    "Chinese Taipei": "Taiwan",
}


def main() -> None:
    print("1/5 Descargando fuentes...")
    download_all()

    print("2/5 Limpiando...")
    former = pd.read_csv(RAW_DIR / "former_names_latest.csv")
    alias = build_alias_map(former)
    matches = clean_results(pd.read_csv(RAW_DIR / "results_latest.csv"), alias)
    shootouts = clean_shootouts(pd.read_csv(RAW_DIR / "shootouts_latest.csv"), alias)

    print("3/5 Cargando a SQLite + Parquet...")
    n = load_matches(matches, shootouts)
    print(f"   {n} partidos cargados")

    print("4/5 Calculando Elo propio...")
    history, ratings = compute_elo(matches)
    history.to_parquet(PROCESSED_DIR / "elo_history.parquet", index=False)
    con = sqlite3.connect(DB_PATH)
    tid = dict(pd.read_sql("SELECT name, team_id FROM teams", con).values)
    h = history.assign(team_id=history["team"].map(tid),
                       date=history["date"].dt.strftime("%Y-%m-%d"))
    con.execute("DELETE FROM elo_history")
    h[["team_id", "date", "elo"]].to_sql("elo_history", con,
                                         if_exists="append", index=False)
    con.commit()

    print("5/5 Rankings FIFA...")
    raw_rank = download_rankings()
    if raw_rank is not None:
        ranks = tidy_rankings(raw_rank, alias_map={**alias, **FIFA_ALIASES})
        ranks = ranks.assign(team_id=ranks["team_name"].map(tid))
        con.execute("DELETE FROM fifa_rankings")
        ranks.to_sql("fifa_rankings", con, if_exists="append", index=False)
        con.commit()
        ranks.to_parquet(PROCESSED_DIR / "fifa_rankings.parquet", index=False)
        print(f"   {len(ranks)} filas de ranking")
    con.close()

    top = sorted(ratings.items(), key=lambda kv: -kv[1])[:10]
    print("\nTop-10 Elo propio:")
    for i, (team, elo) in enumerate(top, 1):
        print(f"  {i:2d}. {team:<15s} {elo:7.1f}")


if __name__ == "__main__":
    main()
