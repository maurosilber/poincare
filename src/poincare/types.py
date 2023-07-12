from __future__ import annotations

from collections import ChainMap
from typing import ClassVar, Generator
from typing import get_type_hints as get_annotations

from symbolite import Scalar, Symbol
from typing_extensions import Self, dataclass_transform, overload


class Constant(Scalar):
    def __init__(self, *, default: Initial):
        self.default = default


Number = int | float | complex
Initial = Number | Constant


class Owned:
    name: str
    parent: System | type[System] | None = None

    def __set_name__(self, cls, name: str):
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "parent", cls)


class ClsMapper(dict):
    def __init__(self, obj: System, cls: type[System]):
        self.obj = obj
        self.cls = cls

    def get(self, item, default):
        if isinstance(item, Owned) and item.parent is self.cls:
            return getattr(self.obj, item.name)
        return item


def derive(
    *,
    variable: Variable,
    order: int,
    initial=None,
    assign=None,
):
    match (initial, assign):
        case (None, None):
            return Derivative(variable=variable, order=order)
        case (initial, None):
            return _create_derivative(
                variable=variable,
                order=order,
                initial=initial,  # type: ignore
                map=0,
            )
        case (None, assign):
            return _assign_equation(
                variable=variable,
                order=order,
                expression=assign,
            )
        case (initial, assign):
            raise ValueError("cannot assign initial and equation.")


def _create_derivative(
    *,
    variable: Variable,
    order: int,
    initial: Initial,
    map: int,
) -> Derivative:
    if not isinstance(initial, Initial):
        raise TypeError(f"unexpected type {type(initial)} for initial")

    if variable.equation_order is not None and order >= variable.equation_order:
        raise ValueError(
            f"already assigned an equation to order {variable.equation_order}"
        )

    if map == 0:
        try:
            value = variable.derivatives.maps[0][order]
            raise ValueError(f"already assigned an initial value: {value}")
        except KeyError:
            variable.derivatives.maps[0][order] = initial
    elif order in variable.derivatives.maps[0]:
        pass
    else:
        try:
            value = variable.derivatives.maps[1][order]
            if value != initial:
                raise ValueError(f"colliding initial value: {value}")
        except KeyError:
            variable.derivatives.maps[1][order] = initial

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
        variable.equation_order = order
        variable.equations.append(expression)
        return Equation(Derivative(variable, order=order), expression)


class Variable(Scalar, Owned):
    derivatives: ChainMap[int, Initial]
    equation_order: int | None = None
    equations: list[Initial | Variable]

    def __str__(self) -> str:
        return str(self.parent) + "." + self.name

    def __init__(self, *, initial: Initial):
        self.derivatives = ChainMap({0: initial}, {})
        self.equations = []

    @property
    def initial(self):
        return self.derivatives[0]

    @overload
    def derive(self, *, initial: None = None, assign: Initial | Variable) -> Equation:
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
        return derive(
            variable=self,
            order=1,
            initial=initial,
            assign=assign,
        )

    def __set__(self, obj, value: Variable | Initial):
        """Allows to override the annotation in System.__init__."""
        # For:
        # >>> class Model(System):
        # ...   x: Variable
        #
        # The type hint shows:
        # >>> Model(x: Variable | Initial) -> None
        if isinstance(value, Variable):
            obj.__dict__[self.name] = value
            for order, initial in self.derivatives.items():
                _create_derivative(
                    variable=value,
                    order=order,
                    initial=initial,
                    map=1,
                )

            if (order := self.equation_order) is not None:
                mapper = ClsMapper(obj, self.parent)
                for expression in self.equations:
                    if isinstance(expression, Symbol):
                        expression = expression.subs(mapper)
                    _assign_equation(
                        variable=value,
                        order=order,
                        expression=expression,
                    )
        elif isinstance(value, Initial):
            variable: Variable = getattr(obj, self.name)
            variable.derivatives[0] = value
            mapper = ClsMapper(obj, self.parent)
            variable.equations = [
                eq.subs(mapper) if isinstance(eq, Symbol) else eq
                for eq in variable.equations
            ]
        else:
            raise TypeError(f"unexpected type {type(value)} for {self.name}")

    def __get__(self, obj, cls) -> Self:
        if obj is None:
            return self

        try:
            return obj.__dict__[self.name]
        except KeyError:
            cls = self.__class__
            copy = cls.__new__(cls)
            copy.__dict__ = self.__dict__.copy()
            copy.derivatives = ChainMap({}, self.derivatives)
            copy.derivatives[0] = self.initial
            copy.parent = obj
            obj.__dict__[self.name] = copy
            return copy

    def __eq__(self, other: Self) -> bool:
        return (self.name == other.name) and (self.initial == other.initial)

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return f"{self.name}={self.initial}"


class Derivative(Variable):
    def __init__(
        self,
        variable: Variable,
        *,
        order: int = 1,
    ):
        self.variable = variable
        self.order = order

    def __get__(self, obj, cls) -> Self:
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
        _create_derivative(
            variable=variable,
            order=self.order,
            initial=value,
            map=0,
        )

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
        return derive(
            variable=self.variable,
            order=self.order + 1,
            initial=initial,
            assign=assign,
        )

    def __lshift__(self, other) -> Equation:
        return _assign_equation(
            variable=self.variable,
            order=self.order,
            expression=other,
        )

    def __eq__(self, other: Self) -> bool:
        if other.__class__ is not self.__class__:
            return NotImplemented

        return self.order == other.order and self.variable == other.variable

    def __repr__(self):
        return f"D({self.variable.name})={self.initial}"


class Equation(Owned):
    def __init__(self, lhs: Derivative, rhs: Initial | Variable):
        self.lhs = lhs
        self.rhs = rhs

    def __get__(self, obj, cls) -> Self:
        if obj is None:
            return self

        try:
            return obj.__dict__[self.name]
        except KeyError:
            equation = Equation(
                lhs=Derivative(
                    getattr(obj, self.lhs.variable.name),
                    order=self.lhs.order,
                ),
                rhs=self.rhs.subs(ClsMapper(obj, cls))
                if isinstance(self.rhs, Symbol)
                else self.rhs,
            )
            obj.__dict__[self.name] = equation
            return equation


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
class System(Owned):
    _kwargs: dict
    _annotations: ClassVar[dict[str, type[Variable | Derivative | System]]]

    def __str__(self) -> str:
        if self.parent is None:
            return getattr(self, "name") or "Root"
        return str(self.parent) + "." + self.name

    def __init_subclass__(cls) -> None:
        cls._annotations = get_annotations(cls)

        for k in ("_annotations", "_kwargs", "name", "parent"):
            del cls._annotations[k]

        # Check mismatched types
        mismatched_types: list[tuple[str, type, type]] = []
        for k, annotation in cls._annotations.items():
            v = getattr(cls, k)
            if not isinstance(v, annotation):
                mismatched_types.append((k, annotation, type(v)))

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

        self._kwargs = kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __set__(self, obj, value):
        if not isinstance(value, self.__class__):
            raise TypeError

        if self.name is None:
            raise RuntimeError
        obj.__dict__[self.name] = value

    def __get__(self, obj, cls) -> Self:
        if obj is None:
            return self

        if self.name is None:
            raise RuntimeError

        try:
            return obj.__dict__[self.name]
        except KeyError:
            kwargs = {}
            for k, v in self._kwargs.items():
                if isinstance(v, Owned) and v.parent is cls:
                    v = getattr(obj, v.name)
                kwargs[k] = v
            copy = self.__class__(**kwargs)
            copy.__set_name__(obj, self.name)
            obj.__dict__[self.name] = copy
            return copy

    def __eq__(self, other: Self):
        """Check equality between Systems.

        Two Systems are equal if they:
        - are instances of the same class
        - have been instanced with the same arguments
        """
        if other.__class__ is not self.__class__:
            return NotImplemented

        return all(getattr(self, k) == getattr(other, k) for k in self._annotations)

    def __repr__(self):
        name = self.__class__.__name__
        kwargs = ",".join(f"{k}={v}" for k, v in self._kwargs.items())
        return f"{name}({kwargs})"

    def yield_variables(
        self, recursive: bool = True
    ) -> Generator[Variable, None, None]:
        for k in self.__class__.__dict__.keys():
            v = getattr(self, k)
            if isinstance(v, System):
                if recursive is True:
                    yield from v.yield_variables(recursive=recursive)
            elif isinstance(v, Variable):
                yield v
