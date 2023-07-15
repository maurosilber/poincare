from collections import defaultdict
from typing import Iterator

from .types import Equation, System, Variable


def yield_equations(system: System | type[System]) -> Iterator[Equation]:
    if isinstance(system, System):
        cls = system.__class__
    else:
        cls = system

    for k, v in cls.__dict__.items():
        if isinstance(v, Equation):
            yield getattr(system, k)
        elif isinstance(v, System):
            yield from yield_equations(getattr(system, k))


def get_equations(system: System | type[System]) -> dict[Variable, list[Equation]]:
    equations: dict[Variable, list[Equation]] = defaultdict(list)
    for eq in yield_equations(system):
        equations[eq.lhs.variable].append(eq)
    return equations


def get_first_order_ode(system: System) -> list:
    equations = get_equations(system)
    return equations
