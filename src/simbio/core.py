from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Iterator, Self, Sequence

from poincare import Constant, Parameter, Variable
from poincare.types import Equation, Owned, System
from symbolite import Symbol
from symbolite.abstract.symbol import BinaryFunction


@dataclass(frozen=True)
class Species(Owned):
    variable: Variable
    stoichiometry: float

    def __str__(self):
        return f"{self.stoichiometry} {self.variable}"

    @classmethod
    def from_symbol(cls, x: Symbol) -> Self:
        if isinstance(x, Variable):
            return cls(variable=x, stoichiometry=1)

        if (
            not isinstance(x, Symbol)
            or x.expression is None
            or not isinstance(x.expression.func, BinaryFunction)
        ):
            raise TypeError("expected a binary expression")

        if x.expression.func.name == "rmul":
            species, stoichiometry = x.expression.args
        elif x.expression.func.name == "mul":
            stoichiometry, species = x.expression.args
        else:
            raise TypeError("expected a multiplication")

        if not isinstance(stoichiometry, int | float):
            raise TypeError("expected a number")

        return cls(variable=species, stoichiometry=stoichiometry)


class Reaction(Owned):
    reactants: Sequence[Species]
    products: Sequence[Species]
    rate_law: Callable
    equations: list[Equation]

    def __init__(
        self,
        *,
        reactants: Sequence[Symbol] = (),
        products: Sequence[Symbol] = (),
        rate_law: Callable,
    ):
        self.reactants = [Species.from_symbol(r) for r in reactants]
        self.products = [Species.from_symbol(p) for p in products]
        self.rate_law = rate_law
        self.equations = list(self.yield_equations())

    def _copy_from(self, parent: System):
        return self.__class__(
            reactants=[
                v.stoichiometry * getattr(parent, v.variable.name)
                for v in self.reactants
            ],
            products=[
                v.stoichiometry * getattr(parent, v.variable.name)
                for v in self.products
            ],
            rate_law=self.rate_law,
        )

    def yield_equations(self) -> Iterator[Equation]:
        rate = self.rate_law(*self.reactants)
        species_stoich: dict[Variable, float] = defaultdict(float)
        for r in self.reactants:
            species_stoich[r.variable] -= r.stoichiometry
        for p in self.products:
            species_stoich[p.variable] += p.stoichiometry

        for s, st in species_stoich.items():
            yield s.derive() << st * rate


class MassAction(Reaction):
    def __init__(
        self,
        *,
        reactants: Sequence[Symbol] = (),
        products: Sequence[Symbol] = (),
        rate: float | Constant | Parameter,
    ):
        self.reactants = [Species.from_symbol(r) for r in reactants]
        self.products = [Species.from_symbol(p) for p in products]
        self.rate = rate
        self.equations = list(self.yield_equations())

    def rate_law(self, *reactants: Species):
        rate = self.rate
        for r in reactants:
            rate *= r.variable**r.stoichiometry
        return rate

    def _copy_from(self, parent: System):
        rate = self.rate
        if isinstance(rate, Owned):
            rate = getattr(parent, rate.name, rate)

        return self.__class__(
            reactants=[
                v.stoichiometry * getattr(parent, v.variable.name)
                for v in self.reactants
            ],
            products=[
                v.stoichiometry * getattr(parent, v.variable.name)
                for v in self.products
            ],
            rate=rate,
        )
