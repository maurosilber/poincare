from __future__ import annotations

from typing_extensions import Self, dataclass_transform


class Variable:
    def __init__(self, *, default: float | None = None):
        # dataclass_transform expects a `default` kw-only parameter.
        self.default = default

    def derive(self, *, default: float | None = None) -> Derivative:
        return Derivative(self, default=default, order=1)

    def integral(self, *, default: float | None = None) -> Derivative:
        return Derivative(self, default=default, order=-1)

    def __set__(self, obj, value: Variable | float):
        # Allows to override the annotation in System.__init__.
        # For:
        # >>> class Model(System):
        # ...   x: Variable
        #
        # The type hint shows:
        # >>> Model(x: Variable | float) -> None
        raise NotImplementedError


class Derivative:
    def __init__(
        self,
        variable: Variable,
        *,
        default: float | None = None,
        order: int,
    ):
        self.variable = variable
        self.order = order
        self.default = default

    def derive(self, *, default: float | None = None) -> Derivative:
        return Derivative(self.variable, default=default, order=self.order + 1)

    def integral(self, *, default: float | None = None) -> Derivative:
        return Derivative(self.variable, default=default, order=self.order + -1)

    def __eq__(self, other: Self):
        if other.__class__ is not self.__class__:
            return NotImplemented

        return self.order == other.order and self.variable == other.variable


@dataclass_transform(field_specifiers=(Variable,))
class System:
    pass
