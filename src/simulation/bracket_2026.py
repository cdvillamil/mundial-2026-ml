"""Cuadro OFICIAL del Mundial 2026 (dieciseisavos P73 -> Final).

Matriz de terceros y arbol de cruces segun el documento oficial de FIFA.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

# Slots de tercero -> grupos permitidos (matriz oficial FIFA)
THIRD_SLOTS = {
    "P74": set("ABCDF"), "P77": set("CDFGH"), "P79": set("CEFHI"),
    "P80": set("EHIJK"), "P81": set("BEFIJ"), "P82": set("AEHIJ"),
    "P85": set("EFGIJ"), "P87": set("DEIJL"),
}

# Dieciseisavos: match -> (slotA, slotB). Slot: ("1",g)=ganador, ("2",g)=segundo,
# ("T",match)=tercero asignado a ese match.
R32 = {
    "P73": (("2", "A"), ("2", "B")),
    "P74": (("1", "E"), ("T", "P74")),
    "P75": (("1", "F"), ("2", "C")),
    "P76": (("1", "C"), ("2", "F")),
    "P77": (("1", "I"), ("T", "P77")),
    "P78": (("2", "E"), ("2", "I")),
    "P79": (("1", "A"), ("T", "P79")),
    "P80": (("1", "L"), ("T", "P80")),
    "P81": (("1", "D"), ("T", "P81")),
    "P82": (("1", "G"), ("T", "P82")),
    "P83": (("2", "K"), ("2", "L")),
    "P84": (("1", "H"), ("2", "J")),
    "P85": (("1", "B"), ("T", "P85")),
    "P86": (("1", "J"), ("2", "H")),
    "P87": (("1", "K"), ("T", "P87")),
    "P88": (("2", "D"), ("2", "G")),
}

# Rondas posteriores: match -> (matchA, matchB) (se enfrentan los ganadores)
LATER = {
    "P89": ("P74", "P77"), "P90": ("P73", "P75"), "P91": ("P76", "P78"),
    "P92": ("P79", "P80"), "P93": ("P83", "P84"), "P94": ("P81", "P82"),
    "P95": ("P86", "P88"), "P96": ("P85", "P87"),
    "P97": ("P89", "P90"), "P98": ("P93", "P94"), "P99": ("P91", "P92"),
    "P100": ("P95", "P96"),
    "P101": ("P97", "P98"), "P102": ("P99", "P100"),
    "FINAL": ("P101", "P102"),
}

# Etapa que alcanza el GANADOR de cada match
_R16 = list(R32.keys())                       # ganar dieciseisavos -> octavos
_QF = ["P89", "P90", "P91", "P92", "P93", "P94", "P95", "P96"]
_SF = ["P97", "P98", "P99", "P100"]
_FINAL = ["P101", "P102"]


def assign_thirds(qualified_groups: set) -> dict:
    """Asigna los 8 grupos-tercero clasificados a los 8 slots respetando la
    matriz oficial. Devuelve {match_id: group}. Matching de costo minimo."""
    slots = list(THIRD_SLOTS)
    groups = sorted(qualified_groups)
    BIG = 1e6
    cost = np.array([[0.0 if g in THIRD_SLOTS[s] else BIG for g in groups]
                     for s in slots])
    row, col = linear_sum_assignment(cost)
    assign = {}
    for r, c in zip(row, col):
        assign[slots[r]] = groups[c]
    for mid, g in assign.items():
        if g not in THIRD_SLOTS[mid]:
            raise ValueError(f"asignacion invalida {mid}<-{g} para {qualified_groups}")
    return assign


def _resolve(slot, winners, runners, thirds_by_match):
    kind, key = slot
    if kind == "1":
        return winners[key]
    if kind == "2":
        return runners[key]
    return thirds_by_match[key]  # ("T", match_id)


def evaluate_bracket(winners: dict, runners: dict, thirds_by_match: dict,
                     decide) -> dict:
    """Evalua el cuadro completo. 'decide(a,b)->ganador'.
    Devuelve {equipo: etapa_mas_alta}."""
    win_cache: dict[str, str] = {}

    def winner(match_id):
        if match_id in win_cache:
            return win_cache[match_id]
        if match_id in R32:
            a = _resolve(R32[match_id][0], winners, runners, thirds_by_match)
            b = _resolve(R32[match_id][1], winners, runners, thirds_by_match)
        else:
            a = winner(LATER[match_id][0])
            b = winner(LATER[match_id][1])
        w = decide(a, b)
        win_cache[match_id] = w
        return w

    winner("FINAL")  # fuerza la evaluacion de todo el arbol

    stages = {}
    for g in winners:
        stages[winners[g]] = "qualify"
        stages[runners[g]] = "qualify"
    for t in thirds_by_match.values():
        stages[t] = "qualify"
    for mid in _R16:
        stages[win_cache[mid]] = "r16"
    for mid in _QF:
        stages[win_cache[mid]] = "qf"
    for mid in _SF:
        stages[win_cache[mid]] = "sf"
    for mid in _FINAL:
        stages[win_cache[mid]] = "final"
    stages[win_cache["FINAL"]] = "champion"
    return stages
