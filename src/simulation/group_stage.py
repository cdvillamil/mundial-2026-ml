"""Fase de grupos: tabla de posiciones con desempates FIFA + mejores terceros."""
import numpy as np


def _standings(teams, results):
    """Calcula puntos, DG, GF por equipo desde los partidos del grupo."""
    table = {t: {"points": 0, "gf": 0, "ga": 0} for t in teams}
    for m in results:
        h, a, hg, ag = m["home"], m["away"], m["hg"], m["ag"]
        if h not in table or a not in table:
            continue
        table[h]["gf"] += hg; table[h]["ga"] += ag
        table[a]["gf"] += ag; table[a]["ga"] += hg
        if hg > ag:
            table[h]["points"] += 3
        elif ag > hg:
            table[a]["points"] += 3
        else:
            table[h]["points"] += 1; table[a]["points"] += 1
    for t in teams:
        table[t]["gd"] = table[t]["gf"] - table[t]["ga"]
    return table


def _h2h_key(group, results):
    """Mini-tabla head-to-head entre los equipos de 'group'."""
    sub = [m for m in results if m["home"] in group and m["away"] in group]
    return _standings(group, sub)


def rank_group(teams, results, rng: np.random.Generator) -> list[str]:
    """Ordena los equipos del grupo segun reglamento FIFA.
    Criterios: puntos -> DG -> GF -> (head-to-head: pts, DG, GF) -> sorteo."""
    st = _standings(teams, results)
    rand = {t: rng.random() for t in teams}

    def primary(t):
        return (-st[t]["points"], -st[t]["gd"], -st[t]["gf"])

    ordered = sorted(teams, key=primary)

    # resolver empates exactos en (pts, DG, GF) con head-to-head, luego sorteo
    result = []
    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and primary(ordered[j + 1]) == primary(ordered[i]):
            j += 1
        tied = ordered[i:j + 1]
        if len(tied) == 1:
            result.append(tied[0])
        else:
            h2h = _h2h_key(tied, results)
            tied_sorted = sorted(
                tied,
                key=lambda t: (-h2h[t]["points"], -h2h[t]["gd"], -h2h[t]["gf"], rand[t])
            )
            result.extend(tied_sorted)
        i = j + 1
    return result


def rank_best_thirds(thirds, n: int, rng: np.random.Generator) -> list[str]:
    """Ordena los terceros (distintos grupos: sin head-to-head) y toma los n mejores."""
    rand = {d["team"]: rng.random() for d in thirds}
    ordered = sorted(
        thirds,
        key=lambda d: (-d["points"], -d["gd"], -d["gf"], rand[d["team"]])
    )
    return [d["team"] for d in ordered[:n]]
