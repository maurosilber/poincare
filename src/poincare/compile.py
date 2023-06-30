from collections import defaultdict

from .types import Equation, System, Variable


def get_equations(system: System) -> dict[Variable, list[Equation]]:
    equations: dict[Variable, list[Equation]] = defaultdict(list)
    for k, v in system.__dict__.items():
        if isinstance(v, Equation):
            equations[v.lhs.variable].append(v)
    return equations


def get_first_order_ode(system: System) -> list:
    equations = get_equations(system)
    return equations


