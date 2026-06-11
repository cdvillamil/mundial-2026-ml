from src.simulation.knockout import propagate_bracket, standard_bracket_order


def test_standard_bracket_order_size_and_seed_separation():
    order = standard_bracket_order(8)
    assert len(order) == 8
    assert set(order) == set(range(1, 9))
    # seed 1 y seed 2 en mitades opuestas (no se cruzan hasta la final)
    assert order.index(1) < 4 and order.index(2) >= 4


def test_propagate_bracket_best_seed_wins_when_deterministic():
    # 4 equipos; el de menor 'seed' siempre gana
    seeds = {"A": 1, "B": 2, "C": 3, "D": 4}
    order = ["A", "D", "C", "B"]  # bracket: A-D, C-B

    def decide(home, away):
        return home if seeds[home] < seeds[away] else away

    rounds = propagate_bracket(order, decide)
    # rounds[-1] es el campeon
    assert rounds[-1] == ["A"]
    # primera ronda: A vence a D; B (seed 2) vence a C (seed 3)
    assert rounds[0] == ["A", "B"]
