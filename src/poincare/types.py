from __future__ import annotations

from typing import ClassVar
from typing import get_type_hints as get_annotations

from symbolite import Scalar
from typing_extensions import Self, dataclass_transform, overload


class Constant(Scalar):
    def __init__(self, *, default: Initial):
        self.default = default


Number = int | float | complex
Initial = Number | Constant


def _create_derivative(
    *,
    variable: Variable,
    order: int,
    initial: Initial,
) -> Derivative:
    if not isinstance(initial, Initial):
        raise TypeError(f"unexpected type {type(initial)} for initial")
    elif variable.equation_order is not None and order >= variable.equation_order:
        raise ValueError(
            f"already assigned an equation to order {variable.equation_order}"
        )
    elif order in variable.derivatives:
        value = variable.derivatives[order]
        raise ValueError(f"already assigned an initial value: {value}")
    else:
        variable.derivatives[order] = initial

    return Derivative(variable, order=order)


def _assign_equation(
    *,
    variable: Variable,
    order: int,
    expression: Initial | Variable,
) -> Equation:
    if variable.equation_order is not None and variable.equation_order != order:
        raise ValueError(
            f"already assigned an equation to order {variable.equation_order}"
        )
    elif order <= (max_order_initial := max(variable.derivatives.keys())):
        raise ValueError(
            f"already assigned an initial to a higher order {max_order_initial}"
        )
    else:
        equation = Equation(Derivative(variable, order=order), expression)
        variable.equation_order = order
        variable.equations.append(equation)
        return equation


class Variable(Scalar):
    name: str
    derivatives: dict[int, Initial]
    equation_order: int | None = None
    equations: list[Equation]

    def __init__(self, *, initial: Initial):
        self.derivatives = {0: initial}
        self.equations = []

    @property
    def initial(self):
        return self.derivatives[0]

    @overload
    def derive(
        self,
        *,
        initial: None = None,
        assign: Initial | Variable,
    ) -> Equation:
        ...

    @overload
    def derive(
        self,
        *,
        initial: Initial | None = None,
        assign: None = None,
    ) -> Derivative:
        ...

    def derive(self, *, initial=None, assign=None):
        variable = self
        order = 1
        match (initial, assign):
            case (None, None):
                return Derivative(variable=variable, order=order)
            case (initial, None):
                return _create_derivative(
                    variable=variable,
                    order=order,
                    initial=initial,  # type: ignore
                )
            case (None, assign):
                return _assign_equation(
                    variable=variable,
                    order=order,
                    expression=assign,
                )
            case (initial, assign):
                raise ValueError("cannot assign initial and equation.")

    def __set_name__(self, obj, name: str):
        object.__setattr__(self, "name", name)

    def __set__(self, obj, value: Variable | Initial):
        """Allows to override the annotation in System.__init__."""
        # For:
        # >>> class Model(System):
        # ...   x: Variable
        #
        # The type hint shows:
        # >>> Model(x: Variable | Initial) -> None
        if isinstance(value, Variable):
            pass
        elif isinstance(value, Initial):
            value = Variable(initial=value)
            value.__set_name__(obj, self.name)
        else:
            raise TypeError(f"unexpected type {type(value)} for {self.name}")

        for order, initial in self.derivatives.items():
            if order not in value.derivatives:
                # We have to differentiate if it was added:
                # - top-level: overrides any value in sub-Systems
                # - lower-levels: set by another sub-System (collison)
                _create_derivative(variable=value, order=order, initial=initial)

        if (order := self.equation_order) is not None:
            for equation in self.equations:
                _assign_equation(
                    variable=value,
                    order=order,
                    expression=equation.rhs,
                )

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

        return Derivative(
            variable=getattr(obj, self.variable.name),
            order=self.order,
        )

    def __set__(self, obj, value: Initial):
        """Allows to override the annotation in System.__init__."""
        # For:
        # >>> class Model(System):
        # ...   x: Derivative
        #
        # The type hint shows:
        # >>> Model(x: Initial) -> None
        if not isinstance(value, Initial):
            raise TypeError(f"expected an initial value for {self.name}")

        variable = getattr(obj, self.variable.name)
        variable.derivatives[self.order] = value

    @property
    def initial(self):
        return self.variable.derivatives[self.order]

    @overload
    def derive(
        self,
        *,
        initial: None = None,
        assign: Initial | Variable,
    ) -> Equation:
        ...

    @overload
    def derive(
        self,
        *,
        initial: Initial | None = None,
        assign: None = None,
    ) -> Derivative:
        ...

    def derive(
        self,
        *,
        initial=None,
        assign=None,
    ):
        variable = self.variable
        order = self.order + 1
        match (initial, assign):
            case None, None:
                return Derivative(variable=variable, order=order)
            case (initial, None):
                return _create_derivative(
                    variable=variable,
                    order=order,
                    initial=initial,  # type: ignore
                )
            case (None, assign):
                return _assign_equation(
                    variable=variable,
                    order=order,
                    expression=assign,
                )
            case initial, assign:
                raise ValueError("cannot assign initial and equation.")

    def __eq__(self, other: Self) -> bool:
        if other.__class__ is not self.__class__:
            return NotImplemented

        return self.order == other.order and self.variable == other.variable

    def __repr__(self):
        return f"D({self.variable.name})={self.initial}"


class Equation:
    def __init__(self, lhs: Derivative, rhs: Initial | Variable):
        self.lhs = lhs
        self.rhs = rhs


@overload
def assign(*, default: Initial) -> Constant:
    ...


@overload
def assign(*, default: Variable) -> Variable:
    ...


def assign(*, default):
    if isinstance(default, Variable):
        return default
    else:
        return Constant(default=default)


def initial(*, default: Initial) -> Variable:
    return Variable(initial=default)


@dataclass_transform(
    kw_only_default=True,
    field_specifiers=(
        initial,
        assign,
    ),
)
class System:
    _annotations: ClassVar[dict[str, type[Variable | Derivative | System]]]
    _required: ClassVar[set[str]]

    def __init_subclass__(cls) -> None:
        cls._annotations = get_annotations(cls)

        for k in ("_annotations", "_required"):
            del cls._annotations[k]

        # Check mismatched types
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
