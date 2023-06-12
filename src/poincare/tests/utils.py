from ..types import Derivative, Variable


def is_same_variable(x: Variable, y: Variable, /) -> bool:
    raise NotImplementedError


def is_derivative(derivative: Derivative, root: Variable) -> bool:
    raise NotImplementedError
