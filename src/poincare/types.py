from __future__ import annotations

from typing import get_type_hints as get_annotations

import symbolite
from symbolite.abstract.symbol import Call
from typing_extensions import Self, dataclass_transform, overload


class Constant(symbolite.scalar.ScalarConstant):
    def __init__(self, *, default: float | None = None):
        self.default = default

    def __lshift__(self, other):
        raise TypeError("unsupported operand")


class Variable(symbolite.Scalar):
    name: str
    derivatives: dict[int, float | None]

    def __init__(self, *, initial: float | Constant | None = None):
        self.initial = initial
        self.derivatives = {}

    def derive(self, *, initial: float | None = None) -> Derivative:
        order = 1
        if self.derivatives.setdefault(order, initial) != initial:
            raise ValueError
        return Derivative(self, order=order)

    def integrate(self, *, initial: float | None = None) -> Derivative:
        raise NotImplementedError

    def __set_name__(self, obj, name: str):
        object.__setattr__(self, "name", name)

    def __set__(self, obj, value: Variable | float):
        """Allows to override the annotation in System.__init__."""
        # For:
        # >>> class Model(System):
        # ...   x: Variable
        #
        # The type hint shows:
        # >>> Model(x: Variable | float) -> None
        if isinstance(value, Variable):
            pass
        elif isinstance(value, (int, float)):
            value = Variable(initial=value)
            value.__set_name__(obj, self.name)
        else:
            raise TypeError(f"unexpected type {type(value)} for {self.name}")

        obj.__dict__[self.name] = value

    def __eq__(self, other: Self) -> bool:
        return (self.name == other.name) and (self.initial == other.initial)

    def __repr__(self):
        return f"{self.name}={self.initial}"


class Derivative(Variable):
    name: str

    def __init__(
        self,
        variable: Variable,
        *,
        order: int = 1,
    ):
        self.variable = variable
        self.order = order

    def __set_name__(self, obj, name: str):
        object.__setattr__(self, "name", name)

    def __get__(self, obj, cls):
        if obj is None:
            return self

        name = self.variable.name
        variable = getattr(obj, name)
        return Derivative(variable, order=self.order)

    def __set__(self, obj, value: float):
        """Allows to override the annotation in System.__init__."""
        # For:
        # >>> class Model(System):
        # ...   x: Derivative
        #
        # The type hint shows:
        # >>> Model(x: float) -> None
        if not isinstance(value, (int, float)):
            raise TypeError(f"expected an initial value for {self.name}")

    @property
    def initial(self):
        return self.variable.derivatives[self.order]

    def derive(self, *, initial: float | None = None) -> Derivative:
        order = self.order + 1
        if self.variable.derivatives.setdefault(order, initial) != initial:
            raise ValueError
        return Derivative(self.variable, order=order)

    def integrate(self, *, initial: float | None = None) -> Derivative:
        raise NotImplementedError

    def __lshift__(self, other: float | Constant | Variable | Call) -> Equation:
        return Equation(self, other)

    def __eq__(self, other: Self) -> bool:
        if other.__class__ is not self.__class__:
            return NotImplemented

        return self.order == other.order and self.variable == other.variable

    def __repr__(self):
        return f"D({self.variable.name})={self.initial}"


class Equation:
    def __init__(self, lhs: Derivative, rhs: float | Constant | Variable | Call):
        self.lhs = lhs
        self.rhs = rhs


@overload
def assign(*, default: float | Constant) -> Constant:
    ...


@overload
def assign(*, default: Variable | Call) -> Call:
    ...


def assign(*, default):
    if isinstance(default, Variable):
        return 1.0 * default
    else:
        return Constant(default=default)


def initial(*, default: float | Constant | None = None) -> Variable:
    return Variable(initial=default)


@dataclass_transform(
    kw_only_default=True,
    field_specifiers=(
        initial,
        assign,
    ),
)
class System:
    _annotations: dict[str, type[Variable | Derivative | System]]
    _required: set[str]

    def __init_subclass__(cls) -> None:
        cls._annotations = get_annotations(cls)

        for k in ("_annotations", "_required"):
            del cls._annotations[k]

        mismatched_types: list[tuple[str, type, type]] = []
        cls._required = set()
        for k, annotation in cls._annotations.items():
            v = getattr(cls, k)
            if not isinstance(v, annotation):
                mismatched_types.append((k, annotation, type(v)))
            if isinstance(v, (Variable, Derivative)) and v.initial is None:
                cls._required.add(k)

        if len(mismatched_types) > 0:
            raise TypeError(
                "\n".join(
                    [f"{k} expected {ann} got {t}" for k, ann, t in mismatched_types]
                )
            )

    def __init__(self, *args, **kwargs):
        if len(args) > 0:
            raise TypeError("positional parameters are not allowed.")

        unexpected = kwargs.keys() - self._annotations.keys()
        if len(unexpected) > 0:
            raise TypeError("unexpected arguments:", unexpected)

        missing = self._required - kwargs.keys()
        if len(missing) > 0:
            raise TypeError("missing parameters:", missing)

        for k, v in kwargs.items():
            setattr(self, k, v)

    def __eq__(self, other: Self):
        if other.__class__ is not self.__class__:
            return NotImplemented

        return all(getattr(self, k) == getattr(other, k) for k in self._annotations)

    def __repr__(self):
        name = self.__class__.__name__
        components = [getattr(self, k) for k in self._annotations]
        components = ", ".join(map(repr, components))
        return f"{name}({components})"


if __name__ == "__main__":

    class Particle(System):
        x: Variable = initial()
        vx: Derivative = x.derive()
        y = Variable(initial=0)
        vy = y.derive(initial=0)

    Particle(x=0, vx=0)
