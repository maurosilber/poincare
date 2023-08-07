from __future__ import annotations

from collections import ChainMap
from typing import ClassVar, Iterator, Literal, Sequence, TypeVar, overload
from typing import get_type_hints as get_annotations

from symbolite import Scalar, Symbol
from typing_extensions import Self, dataclass_transform

from ._node import Node, NodeMapper
from ._utils import class_and_instance_method

T = TypeVar("T")

SimulationTime = Scalar("SimulationTime", namespace="poincare")


class Constant(Node, Scalar):
    def __init__(self, *, default: Initial | None):
        self.default = default

    def _copy_from(self, parent: System):
        return self.__class__(default=NodeMapper(parent).get(self.default))

    def __eq__(self, other: Self):
        if other.__class__ is not self.__class__:
            return NotImplemented

        return self.default == other.default and super().__eq__(other)

    def __hash__(self) -> int:
        return super().__hash__()

    def __set__(self, obj, value: Initial | Constant):
        if isinstance(value, Constant):
            super().__set__(obj, value)
        elif isinstance(value, Initial):
            # Get or create instance with getattr
            constant: Constant = getattr(obj, self.name)
            constant.default = value
        else:
            raise TypeError(f"unexpected type {type(value)} for {self.name}")


Number = int | float | complex
Initial = Number | Constant


def derive(*, variable: Variable, order: int, initial: Initial | None = None):
    if initial is None:
        return Derivative(variable=variable, order=order)
    else:
        return _create_derivative(
            variable=variable,
            order=order,
            initial=initial,  # type: ignore
            map=0,
        )


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


def _assign_equation_order(
    *,
    variable: Variable,
    order: int,
):
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


class Parameter(Node, Scalar):
    equation_order: int = 0

    def __init__(self, *, default: Initial | Symbol | None):
        self.default = default

    def _copy_from(self, parent: System):
        return self.__class__(default=NodeMapper(parent).get(self.default))

    def derive(self, order: int = 1):
        return derive(variable=self, order=1, initial=None)

    def __set__(self, obj, value: Initial | Symbol):
        if isinstance(value, Symbol):
            # Replace descriptor by adding it to obj.__dict__
            # Update derivatives initials
            super().__set__(obj, value)
        elif isinstance(value, Initial):
            # Get or create instance with getattr
            # Update initial value
            variable: Self = getattr(obj, self.name)
            variable.default = value
        else:
            raise TypeError(f"unexpected type {type(value)} for {self.name}")

    def __hash__(self) -> int:
        return super().__hash__()

    def __eq__(self, other: Self):
        if other.__class__ is not self.__class__:
            return NotImplemented

        return self.default == other.default and super().__eq__(other)


class Variable(Parameter):
    derivatives: ChainMap[int, Initial | None]
    equation_order: int | None = None

    def __init__(self, *, initial: Initial | None):
        self.derivatives = ChainMap({0: initial}, {})
        self._equations: list[Equation] = []

    @property
    def initial(self):
        return self.derivatives[0]

    def _copy_from(self, parent: System):
        copy = self.__class__(
            initial=NodeMapper(parent).get(self.initial)
        )  # should initial be here or in maps[1]?
        copy.derivatives.maps[1] = self.derivatives
        copy.equation_order = self.equation_order
        return copy

    def derive(self, *, initial: Initial | None = None) -> Derivative:
        return derive(variable=self, order=1, initial=initial)

    def __set__(self, obj, value: Variable | Initial):
        """Allows to override the annotation in System.__init__."""
        # For:
        # >>> class Model(System):
        # ...   x: Variable
        #
        # The type hint shows:
        # >>> Model(x: Variable | Initial) -> None
        if isinstance(value, Variable):
            # Replace descriptor by adding it to obj.__dict__
            # Update derivatives initials
            super().__set__(obj, value)
            for order, initial in self.derivatives.items():
                _create_derivative(
                    variable=value,
                    order=order,
                    initial=initial,
                    map=1,
                )
            if (order := self.equation_order) is not None:
                _assign_equation_order(variable=value, order=order)
        elif isinstance(value, Initial):
            # Get or create instance with getattr
            # Update initial value
            variable: Variable = getattr(obj, self.name)
            variable.derivatives[0] = value
        else:
            raise TypeError(f"unexpected type {type(value)} for {self.name}")

    def __hash__(self) -> int:
        return super().__hash__()

    def __eq__(self, other: Self):
        if other.__class__ is not self.__class__:
            return NotImplemented

        return self.initial == other.initial and super().__eq__(other)


class Derivative(Node, Symbol):
    def __init__(
        self,
        variable: Variable,
        *,
        order: int = 1,
    ):
        self.variable = variable
        self.order = order

    def __str__(self):
        if self.order == 0:
            return f"{self.variable}"
        return f"{self.variable}.{self.order}"

    def __get__(self, obj, cls) -> Self:
        """Overrides Owned descriptor, as Derivative behaves as a wrapper for Variable."""

        if obj is None:
            return self

        # Get or create the instance variable
        variable: Variable = getattr(obj, self.variable.name)
        return Derivative(variable=variable, order=self.order)

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

        # Get or create the instance variable
        variable: Variable = getattr(obj, self.variable.name)
        _create_derivative(
            variable=variable,
            order=self.order,
            initial=value,
            map=0,
        )

    @property
    def initial(self):
        return self.variable.derivatives[self.order]

    def derive(self, *, initial: Initial | None = None) -> Derivative:
        return derive(variable=self.variable, order=self.order + 1, initial=initial)

    def __lshift__(self, other) -> Equation:
        _assign_equation_order(variable=self.variable, order=self.order)
        return Equation(Derivative(self.variable, order=self.order), other)

    def __hash__(self) -> int:
        return hash((self.variable, self.order))

    def __eq__(self, other: Self) -> bool:
        if other.__class__ is not self.__class__:
            return NotImplemented

        return self.order == other.order and self.variable == other.variable

    def __repr__(self):
        return f"D({self.variable.name}, {self.order})"


class Equation(Node):
    def __init__(self, lhs: Derivative, rhs: Initial | Symbol):
        self.lhs = lhs
        self.rhs = rhs
        # self.lhs.variable._equations.append(self)

    def _copy_from(self, parent: System):
        variable = getattr(parent, self.lhs.variable.name)
        if isinstance(self.rhs, Symbol):
            rhs = self.rhs.subs(NodeMapper(parent))
        else:
            rhs = self.rhs
        return self.__class__(Derivative(variable, order=self.lhs.order), rhs)

    def __repr__(self):
        return f"Equation({self.lhs} << {self.rhs})"


class EquationGroup(Node):
    equations: Sequence[Equation]

    def __init__(self, *equations: Equation):
        self.equations = equations

    def _copy_from(self, parent: System):
        return self.__class__(*(eq._copy_from(parent) for eq in self.equations))


@overload
def assign(
    *,
    default: Initial | Symbol | None = None,
    constant: Literal[False] = False,
    init: bool = True,
) -> Parameter:
    ...


@overload
def assign(
    *,
    default: Initial | None = None,
    constant: Literal[True],
    init: bool = True,
) -> Constant:
    ...


def assign(*, default=None, constant: bool = False, init: bool = True):
    if constant:
        return Constant(default=default)
    else:
        return Parameter(default=default)


def initial(*, default: Initial | None = None, init: bool = True) -> Variable:
    return Variable(initial=default)


class OwnedNamerDict(dict):
    def __setitem__(self, key, value):
        if isinstance(value, Node):
            value.__set_name__(None, key)
        return super().__setitem__(key, value)


class EagerNamer(type):
    @classmethod
    def __prepare__(cls, name, bases):
        return OwnedNamerDict()

    def __str__(self):
        return ""

    def __repr__(self):
        return f"<{self.__name__}>"


@dataclass_transform(
    kw_only_default=True,
    field_specifiers=(
        initial,
        assign,
    ),
)
class System(Node, metaclass=EagerNamer):
    _kwargs: dict
    _required: ClassVar[set[str]]
    _annotations: ClassVar[dict[str, type[Variable | Derivative | System]]]

    simulation_time = SimulationTime

    def __init_subclass__(cls) -> None:
        cls._annotations = get_annotations(cls)

        for k in ("_annotations", "_required", "_kwargs", "name", "parent"):
            del cls._annotations[k]

        # Check mismatched types and compute required
        cls._required = set()
        mismatched_types: list[tuple[str, type, type]] = []
        for k, annotation in cls._annotations.items():
            v = getattr(cls, k)
            if not isinstance(v, annotation):
                mismatched_types.append((k, annotation, type(v)))

            if isinstance(v, Variable) and v.initial is None:
                cls._required.add(k)
            elif isinstance(v, Parameter | Constant) and v.default is None:
                cls._required.add(k)

        if len(mismatched_types) > 0:
            raise TypeError(
                "\n".join(
                    [f"{k} expected {ann} got {t}" for k, ann, t in mismatched_types]
                )
            )

        # for v in cls.yield_variables(cls, recursive=False):
        # for eq in v._equations:
        # if eq.parent is not cls:
        #     raise NameError
        # v._equations.clear()

    def __init__(self, *args, **kwargs):
        if len(args) > 0:
            raise TypeError("positional parameters are not allowed.")

        missing = self._required - kwargs.keys()
        if len(missing) > 0:
            raise TypeError("missing arguments:", missing)

        unexpected = kwargs.keys() - self._annotations.keys()
        if len(unexpected) > 0:
            raise TypeError("unexpected arguments:", unexpected)

        self._kwargs = kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

    def _copy_from(self, parent: System):
        # Create a new instance by replacing previous arguments,
        # which were saved in self._kwargs,
        # with the ones from the corresponding instance
        mapper = NodeMapper(parent)
        kwargs = {k: mapper.get(v, v) for k, v in self._kwargs.items()}
        return self.__class__(**kwargs)

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

    @class_and_instance_method
    def yield_variables(self, *, recursive: bool = True):
        return self._yield(Variable, recursive=recursive)

    @class_and_instance_method
    def yield_equations(self, *, recurisve: bool = True) -> Iterator[Equation]:
        for v in self._yield(Equation | EquationGroup, recursive=recurisve):
            if isinstance(v, Equation):
                yield v
            elif isinstance(v, EquationGroup):
                yield from v.equations
