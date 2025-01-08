from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Generic, ParamSpec, Sequence, TypeVar

if TYPE_CHECKING:
    from poincare import Derivative, Variable

P = ParamSpec("P")
T = TypeVar("T")


@dataclass
class Function(Generic[P, T]):
    func: Callable[P, T]

    def using(self, *args: P.args, **kwds: P.kwargs) -> Call:
        return Call(self.func, args, kwds)


@dataclass
class Call:
    func: Callable
    args: tuple
    kwargs: dict
    output: Sequence[Variable | Derivative] = field(init=False)

    def __rlshift__(self, other: Sequence[Derivative], /):
        self.output = other
        return self
