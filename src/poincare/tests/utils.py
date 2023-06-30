from typing import TypeVar

from ..types import Derivative, Variable

T = TypeVar("T", Variable, Derivative)


def is_same_variable(x: T, y: T, /) -> bool:
    return x == y


def is_derivative(derivative: Derivative, root: Variable) -> bool:
    while isinstance(derivative, Derivative):
        derivative = derivative.variable  # type: ignore
    return derivative == root
