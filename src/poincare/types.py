from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal, Sequence, TypeVar, overload
from typing import get_type_hints as get_annotations

import pint
from symbolite import Scalar, Symbol
from symbolite.core import evaluate
from symbolite.impl import libstd
from typing_extensions import Self, dataclass_transform

from ._node import Node, NodeMapper
from .units import register_with_pint

T = TypeVar("T")


register_with_pint(Symbol)


def check_equations_units(lhs: Derivative, rhs):
    order = 0
    if (value := lhs.variable.initial) is None:
        # Maybe a derivative has a unit already assigned
        for order, der in lhs.variable.derivatives.items():
            if (value := der.initial) is not None:
                break
        else:
            # No unit assigned. Only check that rhs is consistent.
            evaluate(rhs)
            return

    value = evaluate(value)
    rhs = evaluate(rhs)
    if rhs is None:
        return
    if isinstance(value, pint.Quantity):
        time = value._REGISTRY.s
    elif isinstance(rhs, pint.Quantity):
        time = rhs._REGISTRY.s
    else:
        return

    value / time ** (order + 1) - rhs  # check units
    return


def check_units(var, value):
    lhs = evaluate(var)
    rhs = evaluate(value)

    if lhs is not None and rhs is not None:
        lhs - rhs  # must have same units
        return


def check_derivative_units(derivative: Derivative, value):
    check_units(derivative, value)
    check_equations_units(derivative, value)


@register_with_pint
class Constant(Node, Scalar):
    def __init__(self, *, default: Initial | None):
        self.default = default
        check_units(self, default)

    def eval(self, libsl=libstd):
        return evaluate(self.default, libsl)

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
            check_units(constant, value)
            constant.default = value
        else:
            raise TypeError(f"unexpected type {type(value)} for {self.name}")


Number = int | float | complex
Initial = Number | Constant | pint.Quantity


def _assign_equation_order(
    *,
    variable: Variable,
    order: int,
):
    if variable.equation_order is not None and variable.equation_order != order:
        raise ValueError(
            f"already assigned an equation to order {variable.equation_order}"
        )

    for der_order, der in variable.derivatives.items():
        if der_order >= order and der.initial is not None:
            raise ValueError(
                f"already assigned an initial to a higher order {der.order}"
            )

    variable.equation_order = order


@register_with_pint
class Parameter(Node, Scalar):
    equation_order: int = 0

    def __init__(self, *, default: Initial | Symbol | None):
        self.default = default
        check_units(self, default)

    def eval(self, libsl=libstd):
        return evaluate(self.default, libsl)

    def _copy_from(self, parent: System):
        return self.__class__(default=NodeMapper(parent).get(self.default))

    def __set__(self, obj, value: Initial | Symbol):
        if isinstance(value, Parameter) and not isinstance(value, Time):
            super().__set__(obj, value)
        elif isinstance(value, Initial | Symbol):
            # Get or create instance with getattr
            # Update initial value
            variable: Self = getattr(obj, self.name)
            check_units(variable, value)
            variable.default = value
        else:
            raise TypeError(f"unexpected type {type(value)} for {self.name}")

    def __hash__(self) -> int:
        return super().__hash__()

    def __eq__(self, other: Self):
        if other.__class__ is not self.__class__:
            return NotImplemented

        return self.default == other.default and super().__eq__(other)


@register_with_pint
class Variable(Node, Scalar):
    initial: Initial | None
    derivatives: dict[int, Derivative]
    equation_order: int | None = None

    def __init__(self, *, initial: Initial | None):
        self.initial = initial
        self.derivatives = {}
        check_units(self, initial)

    def eval(self, libsl=libstd):
        return evaluate(self.initial, libsl)

    def derive(self, *, initial: Initial | None = None) -> Derivative:
        return Derivative(self, initial=initial, order=1)

    def _copy_from(self, parent: System):
        mapper = NodeMapper(parent)
        copy = self.__class__(initial=mapper.get(self.initial))
        for order, der in self.derivatives.items():
            copy.derivatives[order] = Derivative(
                copy, initial=mapper.get(der.initial), order=order
            )
        copy.equation_order = self.equation_order
        return copy

    def __set__(self, obj, value: Variable | Initial):
        if not isinstance(value, Initial | Variable):
            raise TypeError("unexpected type")

        try:
            variable = obj.__dict__[self.name]
        except KeyError:
            # Variable not yet created
            if isinstance(value, Variable):
                # Replace descriptor by adding it to obj.__dict__
                # Update derivatives initials
                super().__set__(obj, value)
                if (order := self.equation_order) is not None:
                    _assign_equation_order(variable=value, order=order)
                for order, self_der in self.derivatives.items():
                    if self_der.name == "":
                        continue  # derivative taken but not assigned
                    try:
                        value_der = value.derivatives[order]
                    except KeyError:
                        raise TypeError(
                            "must explicitly define higher order derivatives in the external Variable"
                        )
                    else:
                        obj.__dict__[self_der.name] = value_der
            elif isinstance(value, Initial):
                # Create internal variable and update its initial
                variable: Variable = getattr(obj, self.name)
                check_units(variable, value)
                variable.initial = value
        else:
            # Variable already assigned (by a Derivative)
            if isinstance(value, Variable):
                if value is not variable:
                    raise TypeError("wrongly assigned variable or derivative")
            elif isinstance(value, Initial):
                if variable.parent is not obj:
                    raise TypeError("cannot assign initial to an external variable")
                variable.initial = value

    def __hash__(self) -> int:
        return super().__hash__()

    def __eq__(self, other: Self):
        if other.__class__ is not self.__class__:
            return NotImplemented

        return self.initial == other.initial and super().__eq__(other)


@register_with_pint
class Derivative(Node, Symbol):
    def __new__(
        cls,
        variable: Variable,
        *,
        initial: Initial | None = None,
        order: int,
    ):
        try:
            der = variable.derivatives[order]
        except KeyError:
            return super().__new__(cls)

        if initial is None:
            return der
        else:
            raise ValueError(
                f"already assigned an initial value to order {order} of variable {variable}"
            )

    def __init__(
        self,
        variable: Variable,
        *,
        initial: Initial | None = None,
        order: int,
    ):
        self.variable = variable
        self.order = order
        self.initial = initial
        check_derivative_units(self, initial)
        self.variable.derivatives[order] = self

    def eval(self, libsl=libstd):
        return evaluate(self.initial, libsl)

    def _copy_from(self, parent: Node) -> Self:
        variable: Variable = getattr(parent, self.variable.name)
        return variable.derivatives[self.order]

    def __set__(self, obj, value: Initial | Derivative):
        if not isinstance(value, Initial | Derivative):
            raise TypeError("unexpected type")

        # 3 cases to check:
        # - 1. variable and derivative are internal (assigned an Initial or using default)
        # - 2. variable and derivative are external (assigned Variable and Derivative)
        # - 3. variable and internal are mixed internal/external -> error
        # The order of assignment matters.

        try:
            derivative: Derivative = obj.__dict__[self.name]
        except KeyError:
            # derivative has not been assigned -> variable has not been assigned
            if isinstance(value, Derivative):
                # it refers to an outside Variable, let's assign that first
                setattr(obj, self.variable.name, value.variable)
            elif isinstance(value, Initial):
                # this creates an internal variable, which will be an error
                # if it is going to be assigned to an outside variable
                derivative: Derivative = getattr(obj, self.name)
                check_derivative_units(derivative, value)
                derivative.initial = value
        else:
            # derivative has been assigned -> variable has been assigned
            if isinstance(value, Initial):
                # existing derivative must be internal to replace its initial value
                if derivative.parent is not obj:
                    raise TypeError(
                        "derivative corresponds to an external variable, cannot change its initial condition here."
                    )
                else:
                    check_derivative_units(derivative, value)
                    derivative.initial = value
            elif isinstance(value, Derivative) and value is not derivative:
                raise TypeError("assigned wrong derivative")

    def derive(self, *, initial: Initial | None = None) -> Derivative:
        if self.name == "":
            raise NameError("must assign this derivative to a variable first")
        return Derivative(self.variable, initial=initial, order=self.order + 1)

    def __lshift__(self, other) -> Equation:
        _assign_equation_order(variable=self.variable, order=self.order)
        return Equation(self, other)

    def __hash__(self) -> int:
        return hash((self.variable, self.order))

    def __eq__(self, other: Self) -> bool:
        if other.__class__ is not self.__class__:
            return NotImplemented

        return self.order == other.order and self.variable == other.variable

    def __repr__(self):
        return f"D({self.variable.name}, {self.order})"


@dataclass
class Equation(Node):
    lhs: Derivative
    rhs: Initial | Symbol

    def __post_init__(self):
        check_equations_units(self.lhs, self.rhs)

    def _copy_from(self, parent: System):
        variable = getattr(parent, self.lhs.variable.name)
        if isinstance(self.rhs, Symbol):
            rhs = self.rhs.subs(NodeMapper(parent))
        else:
            rhs = self.rhs
        return self.__class__(Derivative(variable, order=self.lhs.order), rhs)

    def __repr__(self):
        return f"Equation({self.lhs} << {self.rhs})"

    def __hash__(self) -> int:
        return super().__hash__()


@dataclass
class EquationGroup(Node):
    equations: Sequence[Equation]

    def __init__(self, *equations: Equation):
        self.equations = equations

    def _copy_from(self, parent: System):
        return self.__class__(*(eq._copy_from(parent) for eq in self.equations))

    def __hash__(self) -> int:
        return super().__hash__()


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


@register_with_pint
class Time(Parameter):
    def _copy_from(self, parent: System):
        raise NotImplementedError

    def __get__(self, obj, cls):
        return self

    def __set__(self, obj, value: Initial | Symbol):
        raise TypeError("cannot modify time")


class OwnedNamerDict(dict):
    def __setitem__(self, key, value):
        if key in self:
            raise NameError(f"duplicate assignment to the same name {key}")
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
    parent: System
    time = Time(default=0)
    _kwargs: dict
    _required: ClassVar[set[str]]
    _annotations: ClassVar[dict[str, type[Variable | Derivative | System]]]

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
