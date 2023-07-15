from collections import defaultdict

from .types import Equation, System, Variable


def get_equations(system: System | type[System]) -> dict[Variable, list[Equation]]:
    if isinstance(system, System):
        eqs = system.yield_equations()
    else:
        eqs = system.yield_equations(system)

    equations: dict[Variable, list[Equation]] = defaultdict(list)
    for eq in eqs:
        equations[eq.lhs.variable].append(eq)
    return equations


def get_first_order_ode(system: System) -> list:
    equations = get_equations(system)
    return equations
