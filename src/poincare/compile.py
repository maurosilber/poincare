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
    Constant,
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


def vector_mapping(
    time: Parameter,
    variables: Sequence[Variable | Derivative],
    parameters: Sequence[Parameter],
    time_varname: str = "t",
    state_varname: str = "y",
    param_varname: str = "p",
) -> dict[Variable | Derivative | Parameter, Symbol]:
    t = scalar.Scalar(time_varname)
    y = vector.Vector(state_varname)
    p = vector.Vector(param_varname)
    mapping: dict[Parameter | Variable | Derivative, Symbol] = {
        time: t,
    }
    for i, v in enumerate(variables):
        mapping[v] = y[i]
    for i, v in enumerate(parameters):
        mapping[v] = p[i]
    return mapping


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
    if not isinstance(value, Symbol):
        return False

    for named in value.yield_named():
        if named is System.simulation_time:
            return True
        elif isinstance(named, Variable | Derivative):
            return True
    return False


def get_derivative(variable: Variable, order: int) -> Variable | Derivative:
    if order == 0:
        return variable
    else:
        return variable.derivatives[order]


def build_equation_maps(system: System | type[System]) -> Compiled[dict]:
    """Compiles equations into dicts of equations.

    - variables: Variable | Derivative
        appears in one or more equations
    - parameters: Parameter
        appears in one or more equations and is not a function of time or variables
    - algebraic_equations: dict[Parameter, RHSExpr]
        parameters that are functions of time or variables
    - differential_equations: dict[Variable | Derivative, RHSExpr]
        variables whose differential are functions of time or variables
    """

    equations = {k: eqsum(v) for k, v in get_equations(system).items()}

    in_eq_parameters: set[Parameter] = set()
    in_eq_variables: set[Variable] = set()

    def process_symbol(symbol):
        if not isinstance(symbol, Symbol):
            return

        for named in symbol.yield_named():
            if isinstance(named, Variable):
                in_eq_variables.add(named)
            elif isinstance(named, Parameter):
                in_eq_parameters.add(named)

    for derivative, eq in equations.items():
        in_eq_variables.add(derivative.variable)
        process_symbol(eq)

    for p in list(in_eq_parameters):
        process_symbol(p.default)

    # Algebraic equations
    # Maps variable to equation.
    parameters = []
    aeqs: dict[Parameter, ExprRHS] = {}
    for p in sorted(in_eq_parameters, key=str):
        if depends_on_at_least_one_variable_or_time(p.default):
            aeqs[p] = p.default
        else:
            parameters.append(p)

    defaults = {}
    for v in system._yield(Constant | Parameter):
        if v.default is None:
            raise TypeError("Missing initial values. System must be instantiated.")
        elif isinstance(v, Parameter) and depends_on_at_least_one_variable_or_time(
            v.default
        ):
            pass
        else:
            defaults[v] = v.default
    for v in system._yield(Variable | Derivative):
        if v.initial is None:
            raise TypeError("Missing initial values. System must be instantiated.")
        defaults[v] = v.initial

    return Compiled(
        variables=sorted(in_eq_variables, key=str),
        parameters=sorted(parameters, key=str),
        mapper=defaults,
        ode_func=equations,
        param_funcs=aeqs,
    )


def build_first_order_symbolic_ode(system: System | type[System]) -> Compiled[dict]:
    maps = build_equation_maps(system)

    # Differential equations
    # Map variable to be derived 1 time to equation.
    # (unlike 'equations' that maps derived variable to equation)
    variables: list[Variable | Derivative] = []
    deqs: dict[Variable | Derivative, ExprRHS] = {}
    for var in maps.variables:
        var: Variable
        # For each variable
        # - create first order differential equations except for var.equation_order
        # - for the var.equation_order use the defined equation
        if var.equation_order is None:
            raise TypeError

        for order in range(var.equation_order - 1):
            lhs = get_derivative(var, order)
            rhs = get_derivative(var, order + 1)
            deqs[lhs] = rhs
            variables.append(lhs)

        order = var.equation_order
        lhs = get_derivative(var, order - 1)
        rhs = get_derivative(var, order)
        deqs[lhs] = maps.ode_func[rhs]
        variables.append(lhs)

    return Compiled[dict](
        variables=variables,
        parameters=maps.parameters,
        mapper=maps.mapper,
        ode_func=deqs,
        param_funcs=maps.param_funcs,
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
    aeqs = symbolic.param_funcs
    deqs = symbolic.ode_func

    mapping = vector_mapping(
        System.simulation_time,
        symbolic.variables,
        symbolic.parameters,
    )
    deqs = {k: substitute(v, ChainMap(aeqs, mapping)) for k, v in deqs.items()}

    def to_index(k: str) -> str:
        try:
            return str(symbolic.variables.index(k))
        except ValueError:
            pass

        try:
            return str(symbolic.parameters.index(k))
        except ValueError:
            pass

        raise ValueError(k)

    ode_step_def = "\n    ".join(
        [
            "def ode_step(t, y, p, ydot):",
            *(assignment_func("ydot", to_index(k), str(eq)) for k, eq in deqs.items()),
            "return ydot",
        ]
    )

    return Compiled(
        variables=symbolic.variables,
        parameters=symbolic.parameters,
        mapper=symbolic.mapper,
        ode_func=ode_step_def,
        param_funcs={
            k: substitute(v, mapping) for k, v in symbolic.param_funcs.items()
        },
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
    lm = symbolite_compile(vectorized.ode_func, libsl)
    return Compiled(
        variables=vectorized.variables,
        parameters=vectorized.parameters,
        mapper=vectorized.mapper,
        ode_func=optimizer(lm["ode_step"]),
        param_funcs=vectorized.param_funcs,  # type: ignore
        libsl=libsl,
    )


@dataclass(frozen=True)
class Compiled(Generic[T]):
    variables: Sequence[Variable | Derivative]
    parameters: Sequence[Parameter]
    mapper: dict[Constant | Variable | Parameter | Derivative, Any]
    ode_func: T
    param_funcs: dict[Parameter, T]
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
