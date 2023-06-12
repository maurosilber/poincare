from __future__ import annotations

from typing_extensions import dataclass_transform


class Variable:
    def __init__(self, *, default: float | None = None):
        # dataclass_transform expects a `default` kw-only parameter.
        raise NotImplementedError

    def derive(self, *, default: float | None = None) -> Derivative:
        raise NotImplementedError

    def __set__(self, obj, value: Variable | float):
        # Allows to override the annotation in System.__init__.
        # For:
        # >>> class Model(System):
        # ...   x: Variable
        #
        # The type hint shows:
        # >>> Model(x: Variable | float) -> None
        raise NotImplementedError


class Derivative(Variable):
    def __init__(self, variable: Variable, *, default: float | None = None):
        self.variable = variable
        self.default = default


@dataclass_transform(field_specifiers=(Variable,))
class System:
    pass
