"""Siembra del bracket y propagacion de eliminatorias."""


def standard_bracket_order(n: int) -> list[int]:
    """Orden de siembra estandar para n = 2^k (1 y 2 en mitades opuestas)."""
    order = [1, 2]
    while len(order) < n:
        size = len(order) * 2
        new = []
        for s in order:
            new.append(s)
            new.append(size + 1 - s)
        order = new
    return order


def seed_qualifiers(ranked_teams: list[str]) -> list[str]:
    """Recibe los clasificados ordenados por calidad (mejor primero) y
    los coloca en posiciones de bracket estandar."""
    n = len(ranked_teams)
    order = standard_bracket_order(n)
    seed_to_team = {seed: ranked_teams[seed - 1] for seed in range(1, n + 1)}
    return [seed_to_team[s] for s in order]


def propagate_bracket(bracket: list[str], decide) -> list[list[str]]:
    """Propaga el cuadro. 'decide(home, away)->ganador'.
    Devuelve lista de rondas: [ganadores_ronda1, ..., [campeon]]."""
    rounds = []
    current = list(bracket)
    while len(current) > 1:
        winners = []
        for k in range(0, len(current), 2):
            winners.append(decide(current[k], current[k + 1]))
        rounds.append(winners)
        current = winners
    return rounds
