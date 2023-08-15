from __future__ import annotations

import enum
from collections import defaultdict
from collections.abc import MutableSequence, Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import (
    Any,
    Callable,
    Generic,
    Iterator,
    Mapping,
    Protocol,
    TypeAlias,
    TypeVar,
)

import symbolite.abstract as libabstract
from symbolite import Symbol, scalar, vector
from symbolite.core import compile as symbolite_compile
from symbolite.core import substitute
from typing_extensions import Never

from ._node import Node
from ._utils import eval_content
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

V = TypeVar("V")
F = TypeVar("F")
ExprRHS: TypeAlias = Initial | Symbol
Array: TypeAlias = Sequence[float]
MutableArray: TypeAlias = MutableSequence[float]


class RHS(Protocol):
    def __call__(self, t: float, y: Array, p: Array, dy: MutableArray) -> Array:
        ...


class Transform(Protocol):
    def __call__(self, t: float, y: Array, p: Array, out: MutableArray) -> Array:
        ...


@dataclass(frozen=True, kw_only=True)
class Compiled(Generic[V, F]):
    variables: Sequence[V]
    parameters: Sequence[Symbol]
    mapper: dict[Symbol, Any]
    func: F
    output_names: Sequence[str]
    libsl: ModuleType | None = None


def identity(x):
    return x


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
    time: scalar.Scalar,
    variables: Sequence[Variable | Derivative],
    parameters: Sequence[Parameter],
    time_varname: str = "t",
    state_varname: str = "y",
    param_varname: str = "p",
) -> dict[scalar.Scalar | Variable | Derivative | Parameter, Symbol]:
    t = scalar.Scalar(time_varname)
    y = vector.Vector(state_varname)
    p = vector.Vector(param_varname)
    mapping: dict[scalar.Scalar | Parameter | Variable | Derivative, Symbol] = {
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
        elif isinstance(named, Parameter) and depends_on_at_least_one_variable_or_time(
            named.default
        ):
            return True
    return False


def get_derivative(variable: Variable, order: int) -> Variable | Derivative:
    if order == 0:
        return variable
    else:
        return variable.derivatives[order]


def build_equation_maps(
    system: System | type[System],
) -> Compiled[Variable, dict[Derivative, ExprRHS]]:
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

    algebraic: dict[Parameter, ExprRHS] = {}
    equations: dict[Derivative, ExprRHS] = {
        k: eqsum(v) for k, v in get_equations(system).items()
    }

    initials = {}
    parameters: set[Parameter] = set()
    variables: set[Variable] = set()

    def add_to_initials(name: Symbol, value):
        if value is None:
            raise TypeError(
                f"Missing initial value for {name}. System must be instantiated."
            )
        initials[name] = value
        if not isinstance(value, Symbol):
            return

        for named in value.yield_named():
            if isinstance(named, Parameter | Constant):
                add_to_initials(named, named.default)

    def process_symbol(symbol):
        if not isinstance(symbol, Symbol):
            return

        for named in symbol.yield_named():
            if isinstance(named, Variable):
                variables.add(named)
                for order in range(named.equation_order):
                    der = get_derivative(named, order)
                    add_to_initials(der, der.initial)
            elif isinstance(
                named, Parameter
            ) and depends_on_at_least_one_variable_or_time(named.default):
                algebraic[named] = named.default
                process_symbol(named.default)
            elif isinstance(named, Constant | Parameter):
                parameters.add(named)
                add_to_initials(named, named.default)

    for derivative, eq in equations.items():
        process_symbol(derivative.variable)
        process_symbol(eq)

    content = {
        **equations,
        **algebraic,
        **{p: p for p in parameters},
        **{v: v for v in variables},
    }
    content = eval_content(
        content,
        libabstract,
        Node,
    )

    equations = {k: content[k] for k in equations.keys()}
    sorted_variables = sorted(variables, key=str)
    return Compiled(
        variables=sorted_variables,
        parameters=sorted(parameters, key=str),
        mapper=initials,
        func=equations,
        output_names=tuple(map(str, sorted_variables)),
    )


def build_first_order_symbolic_ode(
    system: System | type[System],
) -> Compiled[Variable | Derivative, dict[Variable | Derivative, ExprRHS]]:
    maps = build_equation_maps(system)

    # Differential equations
    # Map variable to be derived 1 time to equation.
    # (unlike 'equations' that maps derived variable to equation)
    variables: list[Variable | Derivative] = []
    diff_eqs: dict[Variable | Derivative, ExprRHS] = {}
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
            diff_eqs[lhs] = rhs
            variables.append(lhs)

        order = var.equation_order
        lhs = get_derivative(var, order - 1)
        rhs = get_derivative(var, order)
        diff_eqs[lhs] = maps.func[rhs]
        variables.append(lhs)

    return Compiled(
        variables=variables,
        output_names=tuple(map(str, variables)),
        func=diff_eqs,
        parameters=maps.parameters,
        mapper=maps.mapper,
    )


def assignment(name: str, index: str, value: str) -> str:
    return f"{name}[{index}] = {value}"


def jax_assignment(name: str, index: str, value: str) -> str:
    return f"{name} = {name}.at[{index}].set({value})"


def build_first_order_vectorized_body(
    system: System | type[System],
    *,
    assignment_func: Callable[[str, str, str], str] = assignment,
) -> Compiled[Variable | Derivative, str]:
    symbolic = build_first_order_symbolic_ode(system)
    mapping: Mapping = vector_mapping(
        System.simulation_time,
        symbolic.variables,
        symbolic.parameters,
    )

    diff_eqs = {k: substitute(v, mapping) for k, v in symbolic.func.items()}

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
            *(
                assignment_func("ydot", to_index(k), str(eq))
                for k, eq in diff_eqs.items()
            ),
            "return ydot",
        ]
    )

    return Compiled(
        variables=symbolic.variables,
        parameters=symbolic.parameters,
        mapper=symbolic.mapper,
        func=ode_step_def,
        output_names=symbolic.output_names,
    )


def build_first_order_functions(
    system: System | type[System],
    libsl: ModuleType,
    optimizer: Callable[[RHS], RHS] = identity,
    assignment_func: Callable[[str, str, str], str] = assignment,
) -> Compiled[Variable | Derivative, RHS]:
    vectorized = build_first_order_vectorized_body(
        system, assignment_func=assignment_func
    )
    lm = symbolite_compile(vectorized.func, libsl)
    return Compiled(
        variables=vectorized.variables,
        parameters=vectorized.parameters,
        mapper=vectorized.mapper,
        func=optimizer(lm["ode_step"]),
        output_names=vectorized.output_names,
        libsl=libsl,
    )


def compile_diffeq(
    system: System | type[System],
    backend: Backend = Backend.FIRST_ORDER_VECTORIZED_NUMPY_NUMBA,
) -> Compiled[Variable | Derivative, RHS]:
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


def identity_transform(t: float, y: Array, p: Array, out: MutableArray) -> Array:
    out[:] = y
    return out


def compile_transform(
    compiled: Compiled,
    expresions: dict[str, Symbol] | None = None,
) -> Compiled[Variable | Derivative, Transform]:
    if expresions is None:
        return Compiled(
            func=identity_transform,
            output_names=compiled.output_names,
            variables=compiled.variables,
            parameters=compiled.parameters,
            mapper=compiled.mapper,
            libsl=compiled.libsl,
        )

    mapping: Mapping = vector_mapping(
        System.simulation_time,
        compiled.variables,
        compiled.parameters,
    )

    def to_index(k: str) -> str:
        try:
            return str(compiled.variables.index(k))
        except ValueError:
            pass

        try:
            return str(compiled.parameters.index(k))
        except ValueError:
            pass

        raise ValueError(k)

    deqs = {k: substitute(v, mapping) for k, v in expresions.items()}
    ode_step_def = "\n    ".join(
        [
            "def transform(t, y, p, out):",
            *(assignment("out", str(i), str(eq)) for i, eq in enumerate(deqs.values())),
            "return out",
        ]
    )
    lm = symbolite_compile(ode_step_def, compiled.libsl)
    return Compiled(
        func=lm["transform"],
        output_names=tuple(map(str, deqs.keys())),
        variables=compiled.variables,
        parameters=compiled.parameters,
        mapper=compiled.mapper,
        libsl=compiled.libsl,
    )
