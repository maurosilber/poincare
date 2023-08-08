from __future__ import annotations

import enum
from collections import ChainMap, defaultdict
from collections.abc import MutableSequence, Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Callable, Generic, Iterator, Protocol, TypeAlias, TypeVar

from symbolite import Symbol, scalar, vector
from symbolite.core import compile as symbolite_compile
from symbolite.core import substitute
from typing_extensions import Never

from .types import (
    Derivative,
    Equation,
    EquationGroup,
    Initial,
    Parameter,
    System,
    Variable,
)

T = TypeVar("T")
ExprRHS: TypeAlias = Initial | Symbol
Array: TypeAlias = Sequence[float]
MutableArray: TypeAlias = MutableSequence[float]


class RHS(Protocol):
    def __call__(self, t: float, y: Array, p: Array, dy: MutableArray) -> Array:
        ...


def identity(fn: RHS) -> RHS:
    return fn


class Backend(enum.Enum):
    FIRST_ORDER_VECTORIZED_STD = enum.auto()
    FIRST_ORDER_VECTORIZED_NUMPY = enum.auto()
    FIRST_ORDER_VECTORIZED_NUMPY_NUMBA = enum.auto()
    FIRST_ORDER_VECTORIZED_JAX = enum.auto()

    @staticmethod
    def get_libsl(backend: Backend) -> ModuleType:
        match backend:
            case Backend.FIRST_ORDER_VECTORIZED_STD:
                from symbolite.impl import libstd

                return libstd
            case Backend.FIRST_ORDER_VECTORIZED_NUMPY:
                from symbolite.impl import libnumpy

                return libnumpy
            case Backend.FIRST_ORDER_VECTORIZED_NUMPY_NUMBA:
                from symbolite.impl import libnumpy

                return libnumpy
            case Backend.FIRST_ORDER_VECTORIZED_JAX:
                from symbolite.impl import libjax

                return libjax
            case _:
                assert_never(backend, message="Unknown backend {}")


def eqsum(eqs: list[ExprRHS]) -> scalar.NumberT | Symbol:
    if len(eqs) == 0:
        return 0
    elif len(eqs) == 1:
        return eqs[0]
    else:
        return sum(eqs[1:], start=eqs[0])


def ode_vectorize(
    expr: scalar.NumberT | Symbol,
    state_names: tuple[str, ...],
    param_names: tuple[str, ...],
    state_varname: str = "y",
    param_varname: str = "p",
) -> scalar.NumberT | Symbol:
    expr = vector.vectorize(
        expr, state_names, varname=state_varname, scalar_type=SimpleVariable
    )
    expr = vector.vectorize(
        expr, param_names, varname=param_varname, scalar_type=SimpleParameter
    )
    return expr


def yield_equations(system: System | type[System]) -> Iterator[Equation]:
    for v in system._yield(Equation | EquationGroup):
        if isinstance(v, Equation):
            yield v
        elif isinstance(v, EquationGroup):
            yield from v.equations
        else:
            assert_never(v, message="unexpected type {}")


def get_equations(system: System | type[System]) -> dict[Derivative, list[ExprRHS]]:
    equations: dict[Derivative, list[ExprRHS]] = defaultdict(list)
    for eq in yield_equations(system):
        equations[eq.lhs].append(eq.rhs)
    return equations


def depends_on_at_least_one_variable_or_time(value: Any) -> bool:
    if not hasattr(value, "yield_named"):
        return False
    for named in value.yield_named():
        if named is System.simulation_time:
            return True
        if isinstance(named, Derivative):
            return True
        elif isinstance(named, Variable):
            return True
    return False


@dataclass(frozen=True)
class SimpleVariable(scalar.Scalar):
    """Special type of Scalar that is evaluated to itself."""

    def __repr__(self):
        return self.name

    def eval(self, libsl: ModuleType | None = None):
        return self


@dataclass(frozen=True)
class SimpleParameter(scalar.Scalar):
    """Special type of Scalar that is evaluated to itself."""

    def __repr__(self):
        return self.name

    def eval(self, libsl: ModuleType | None = None):
        return self


def build_first_order_symbolic_ode(system: System | type[System]):
    parameters: dict[Parameter, SimpleParameter] = {}
    variables: dict[Variable, SimpleVariable] = {}
    derivatives: dict[Derivative, SimpleVariable] = {}
    for v in system._yield(Parameter | Variable | Derivative):
        match v:
            case Derivative(variable=var, order=order):
                derivatives[v] = SimpleVariable(f"{var}.{order}")
            case Variable():
                variables[v] = SimpleVariable(f"{v}.0")
            case Parameter():
                parameters[v] = SimpleParameter(str(v))

    mapper = ChainMap(variables, derivatives, parameters)
    equations = {k: eqsum(v) for k, v in get_equations(system).items()}

    # Algebraic equations
    # Maps variable to equation.
    aeqs: dict[SimpleVariable, Any] = {
        # TODO: this is wrong. Should not get k.default but rather get it from
        # Equations??
        substitute(k, mapper): substitute(k.default, mapper)
        for k in parameters
        if depends_on_at_least_one_variable_or_time(k.default)
    }

    # Differential equations
    # Map variable to be derived 1 time to equation.
    # (unlike 'equations' that maps derived variable to equation)
    deqs: dict[SimpleVariable, Any] = {}

    for var in variables:
        # For each variable
        # - create first order differential equations except for var.equation_order
        # - for the var.equation_order use the defined equation
        if var.equation_order is None:
            continue

        for order in range(1, var.equation_order):
            lhs = SimpleVariable(f"{var}.{order-1}")
            rhs = SimpleVariable(f"{var}.{order}")
            deqs[lhs] = rhs

        lhs = SimpleVariable(f"{var}.{var.equation_order - 1 }")
        rhs = substitute(equations[Derivative(var, order=var.equation_order)], mapper)
        deqs[lhs] = rhs

    # The state variables are all the keys in the differential equations
    # TODO: add algebraic equations
    state_variables = tuple(deqs.keys())

    return Compiled[dict](
        variable_names=state_variables,
        parameter_names=tuple(substitute(k, mapper) for k in parameters),
        ode_func=deqs,
        param_func=aeqs,
        mapper=mapper,
    )


def assignment(name: str, index: str, value: str) -> str:
    return f"{name}[{index}] = {value}"


def jax_assignment(name: str, index: str, value: str) -> str:
    return f"{name} = {name}.at[{index}].set({value})"


def build_first_order_vectorized_body(
    system: System | type[System],
    *,
    assignment_func: Callable[[str, str, str], str] = assignment,
) -> Compiled[str]:
    symbolic = build_first_order_symbolic_ode(system)
    aeqs = symbolic.param_func
    deqs = symbolic.ode_func
    state_variables = symbolic.variable_names
    parameters = symbolic.parameter_names
    mapper = symbolic.mapper

    state_names = tuple(sorted(str(v) for v in state_variables))
    param_names = tuple(sorted(str(p) for p in parameters))

    aeqs = {str(k): ode_vectorize(v, state_names, param_names) for k, v in aeqs.items()}
    deqs = {str(k): ode_vectorize(v, state_names, param_names) for k, v in deqs.items()}

    def to_index(k: str) -> str:
        try:
            return str(state_names.index(k))
        except ValueError:
            pass

        try:
            return str(param_names.index(k))
        except ValueError:
            pass

        raise ValueError(k)

    update_param_body = [
        assignment_func("p", to_index(k), str(eq)) for k, eq in aeqs.items()
    ]
    ode_step_body = [
        assignment_func("ydot", to_index(k), str(eq)) for k, eq in deqs.items()
    ]

    update_param_def = "\n    ".join(
        [
            "def update_param(t, y, p0, p):",
            *update_param_body,
            "return p",
        ]
    )

    ode_step_def = "\n    ".join(
        [
            "def ode_step(t, y, p, ydot):",
            *update_param_body,
            *ode_step_body,
            "return ydot",
        ]
    )

    inverse = {str(v): str(k) for k, v in mapper.items()}  # TODO: repeated v?
    return Compiled(
        tuple(inverse[k] for k in state_names),
        tuple(inverse[k] for k in param_names),
        ode_step_def,
        update_param_def,
        symbolic.mapper,
    )


def build_first_order_functions(
    system: System | type[System],
    libsl: ModuleType,
    optimizer: Callable[[RHS], RHS] = identity,
    assignment_func: Callable[[str, str, str], str] = assignment,
) -> Compiled[RHS]:
    vectorized = build_first_order_vectorized_body(
        system, assignment_func=assignment_func
    )

    lm = symbolite_compile(
        "\n".join(
            [
                vectorized.ode_func,
                vectorized.param_func,
            ]
        ),
        libsl,
    )

    return Compiled(
        vectorized.variable_names,
        vectorized.parameter_names,
        optimizer(lm["ode_step"]),
        optimizer(lm["update_param"]),
        vectorized.mapper,
        libsl,
    )


@dataclass(frozen=True)
class Compiled(Generic[T]):
    variable_names: tuple[str, ...]
    parameter_names: tuple[str, ...]
    ode_func: T
    param_func: T
    mapper: dict
    libsl: ModuleType | None = None


def compile(
    system: System | type[System],
    backend: Backend = Backend.FIRST_ORDER_VECTORIZED_NUMPY_NUMBA,
) -> Compiled[RHS]:
    libsl = Backend.get_libsl(backend)

    optimizer_fun = identity
    assignment_fun = assignment
    match backend:
        case Backend.FIRST_ORDER_VECTORIZED_NUMPY_NUMBA:
            import numba

            optimizer_fun = numba.njit
        case Backend.FIRST_ORDER_VECTORIZED_JAX:
            import jax

            optimizer_fun = jax.jit
            assignment_fun = jax_assignment

    return build_first_order_functions(system, libsl, optimizer_fun, assignment_fun)


def assert_never(arg: Never, *, message: str) -> Never:
    raise ValueError(message.format(arg))
