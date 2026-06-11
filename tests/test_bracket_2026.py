from src.simulation.bracket_2026 import THIRD_SLOTS, assign_thirds, evaluate_bracket


def test_assign_thirds_respects_allowed_groups():
    qual = set("ABCDEFGH")
    assign = assign_thirds(qual)
    assert len(assign) == 8
    for match_id, group in assign.items():
        assert group in THIRD_SLOTS[match_id]
    assert len(set(assign.values())) == 8


def test_assign_thirds_handles_another_combo():
    qual = set("CEFHIJKL")
    assign = assign_thirds(qual)
    assert len(assign) == 8
    assert set(assign.values()) == qual


def test_evaluate_bracket_produces_one_champion():
    letters = list("ABCDEFGHIJKL")
    winners = {g: f"W{g}" for g in letters}
    runners = {g: f"R{g}" for g in letters}
    thirds_team = {g: f"T{g}" for g in "ABCDEFGH"}
    assign = assign_thirds(set("ABCDEFGH"))
    thirds_by_match = {mid: thirds_team[grp] for mid, grp in assign.items()}

    strength = {}
    for g in letters:
        strength[f"W{g}"] = 3
        strength[f"R{g}"] = 2
    for g in "ABCDEFGH":
        strength[f"T{g}"] = 1

    def decide(a, b):
        return a if strength.get(a, 0) >= strength.get(b, 0) else b

    stages = evaluate_bracket(winners, runners, thirds_by_match, decide)
    champions = [t for t, s in stages.items() if s == "champion"]
    assert len(champions) == 1
    assert len(stages) == 32
