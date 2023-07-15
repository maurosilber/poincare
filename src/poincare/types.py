from __future__ import annotations

from collections import ChainMap
from typing import ClassVar, Iterator
from typing import get_type_hints as get_annotations

from symbolite import Scalar, Symbol
from typing_extensions import Self, dataclass_transform, overload


class OwnedNamerDict(dict):
    def __setitem__(self, key, value):
        if isinstance(value, Owned):
            value.__set_name__(None, key)
        return super().__setitem__(key, value)


class EagerNamer(type):
    @classmethod
    def __prepare__(cls, name, bases):
        return OwnedNamerDict()

    def __repr__(self):
        return f"<{self.__name__}>"


class Owned:
    name: str = ""
    parent: System | type[System] | None = None

    def __set_name__(self, cls, name: str):
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "parent", cls)

    def __set__(self, obj: System, value: Self):
        if not isinstance(value, self.__class__):
            raise TypeError

        if value.parent is None and value.name == "":
            # if it has no name, it was created outside an EagerNamer
            value.__set_name__(obj, self.name)

        obj.__dict__[self.name] = value

    def __str__(self) -> str:
        return f"{self.parent}.{self.name}"

    def __repr__(self):
        return str(self)

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other):
        if other.__class__ is not self.__class__:
            return NotImplemented

        return str(self) == str(other)


class Constant(Owned, Scalar):
    def __init__(self, *, default: Initial):
        self.default = default

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

    def __get__(self, obj, cls):
        if obj is None:
            return self

        try:
            return obj.__dict__[self.name]
        except KeyError:
            value = Constant(default=self.default)
            super().__set__(obj, value)
            return value


Number = int | float | complex
Initial = Number | Constant


class ClsMapper(dict):
    def __init__(self, obj: System, cls: type[System]):
        self.obj = obj
        self.cls = cls

    def get(self, item, default):
        if not isinstance(item, Owned):
            return item

        # Recursively look all parents
        # If an item's parent is cls,
        #   replace that item for the instance's corresponding item.
        #   go down the name list to fetch the original item, adn return that
        # else return the original item as is.
        default = item
        names = []
        while item.parent is not None:
            names.append(item.name)
            item = item.parent
            if item is self.cls:
                item = self.obj
                for name in reversed(names):
                    item = getattr(item, name)
                return item
        else:
            return default


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
            _assign_equation_order(variable=variable, order=order)
            return Equation(Derivative(variable, order=order), assign)
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


class Variable(Owned, Scalar):
    derivatives: ChainMap[int, Initial]
    equation_order: int | None = None

    def __init__(self, *, initial: Initial):
        self.derivatives = ChainMap({0: initial}, {})

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

    def __get__(self, obj, cls) -> Self:
        if obj is None:
            return self

        try:
            return obj.__dict__[self.name]
        except KeyError:
            # Create new instance by copying descriptor.
            cls = self.__class__
            copy = cls.__new__(cls)
            # Set name and parent and save in instance.__dict__ for future access
            super().__set__(obj, copy)
            # Set descriptor derivatives as default derivatives
            copy.derivatives = ChainMap({0: self.initial}, self.derivatives)
            copy.equation_order = self.equation_order
            return copy

    def __hash__(self) -> int:
        return super().__hash__()

    def __eq__(self, other):
        if other.__class__ is not self.__class__:
            return NotImplemented

        return self.initial == other.initial and super().__eq__(other)


class Derivative(Variable):
    def __init__(
        self,
        variable: Variable,
        *,
        order: int = 1,
    ):
        self.variable = variable
        self.order = order

    def __str__(self):
        return f"{self.variable}.{self.order}"

    def __get__(self, obj, cls) -> Self:
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
        _assign_equation_order(variable=self.variable, order=self.order)
        return Equation(Derivative(self.variable, order=self.order), other)

    def __eq__(self, other: Self) -> bool:
        if other.__class__ is not self.__class__:
            return NotImplemented

        return self.order == other.order and self.variable == other.variable

    def __repr__(self):
        return f"D({self.variable.name})"


class Equation(Owned):
    def __init__(self, lhs: Derivative, rhs: Initial | Variable):
        self.lhs = lhs
        self.rhs = rhs

    def __repr__(self):
        return f"Equation({self.lhs} << {self.rhs})"

    def __get__(self, obj, cls) -> Self:
        if obj is None:
            return self

        try:
            return obj.__dict__[self.name]
        except KeyError:
            # Recreate the equation by replacing all variables with instance variables.
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
class System(Owned, metaclass=EagerNamer):
    _kwargs: dict
    _annotations: ClassVar[dict[str, type[Variable | Derivative | System]]]

    def __str__(self) -> str:
        if self.parent is None:
            return getattr(self, "name", "Root")
        return f"{self.parent}.{self.name}"

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

    def __get__(self, obj, cls) -> Self:
        if obj is None:
            return self

        if self.name is None:
            raise RuntimeError

        try:
            return obj.__dict__[self.name]
        except KeyError:
            # Create a new instance by replacing previous arguments,
            # which were saved in self._kwargs,
            # with the ones from the corresponding instance
            kwargs = {
                k: getattr(obj, v.name)
                if isinstance(v, Owned) and v.parent is cls
                else v
                for k, v in self._kwargs.items()
            }
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

    def yield_variables(self, recursive: bool = True) -> Iterator[Variable]:
        for k in self.__class__.__dict__.keys():
            v = getattr(self, k)
            if isinstance(v, System):
                if recursive is True:
                    yield from v.yield_variables(recursive=recursive)
            elif isinstance(v, Variable):
                yield v

    def yield_equations(self: Self | type[Self]) -> Iterator[Equation]:
        if isinstance(self, System):
            cls = self.__class__
        else:
            cls = self

        for k, v in cls.__dict__.items():
            if isinstance(v, Equation):
                yield getattr(self, k)
            elif isinstance(v, System):
                v: System = getattr(self, k)
                yield from v.yield_equations()
