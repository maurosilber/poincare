from __future__ import annotations

import enum
from collections import defaultdict
from collections.abc import MutableSequence, Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Callable, Generic, Protocol, TypeAlias, TypeVar

from symbolite import Symbol, scalar, vector
from symbolite.core import compile as symbolite_compile
from symbolite.core import substitute
from typing_extensions import Never

from .types import Constant, Derivative, Initial, Parameter, System, Variable

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


def get_equations(system: System | type[System]) -> dict[Derivative, list[ExprRHS]]:
    equations: dict[Derivative, list[ExprRHS]] = defaultdict(list)
    for eq in system.yield_equations():
        equations[eq.lhs].append(eq.rhs)
    return equations


def get_initial_values(system: System | type[System]) -> dict[Derivative, Initial]:
    initial_values: dict[Derivative, Initial] = {}
    for var in system.yield_variables():
        for order, initial_value in var.derivatives.items():
            initial_values[Derivative(var, order=order)] = initial_value
    return initial_values


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

    def eval(self, libsl: ModuleType | None = None):
        return self


@dataclass(frozen=True)
class SimpleParameter(scalar.Scalar):
    """Special type of Scalar that is evaluated to itself."""

    def eval(self, libsl: ModuleType | None = None):
        return self


class SystemContentMapper(dict[Any, Any]):
    """Used to convert Poincare Variables into SelfEvalScalar.

    Benefetis:
    - leaner
    - immutable
    - understood by Symbolite to vectorize
    - can be involved when using evaluate to simplify part of the code.
    """

    def __init__(
        self,
        variables: tuple[Variable, ...],
        parameters: tuple[Variable | Constant, ...],
    ):
        self.variables = variables
        self.parameters = parameters

    def get(self, key: Any, default=None):
        if key is System.simulation_time:
            return SimpleParameter("t")
        elif isinstance(key, (Derivative, Variable)):
            if isinstance(key, Derivative):
                assert isinstance(key.variable, Variable)
                if key.variable in self.variables:
                    return SimpleVariable(str(key))
            elif key in self.parameters:
                return SimpleParameter(str(key))
            elif key in self.variables:
                return SimpleVariable(str(key))
            else:
                raise ValueError(f"Not found {key}")
        elif isinstance(key, (Constant, Parameter)):
            return SimpleParameter(str(key))
        return key


def build_first_order_symbolic_ode(
    system: System | type[System],
) -> tuple[
    dict[SimpleVariable, scalar.NumberT | Symbol],
    dict[SimpleVariable, scalar.NumberT | Symbol],
    dict[SimpleVariable, scalar.NumberT | Symbol],
    tuple[SimpleVariable, ...],
    tuple[SimpleParameter, ...],
]:
    #############
    # Step 0:
    # Get initial value and flattened equations

    initial_values = get_initial_values(system)
    equations = {k: eqsum(v) for k, v in get_equations(system).items()}

    #############
    # Step 1:
    # Divide between:
    # - parameters: variables that appear with in a single "order" and constants.
    # - variables: variables appear with more than one "order".

    variables: set[Variable] = set()
    parameters: set[Constant | Variable] = set()

    # Create an inventory of all variables in the systems
    # taking note in which their derivative
    # Either in the LHS of equations and initial values
    inventory: defaultdict[Variable, set[int]] = defaultdict(set)
    for var in tuple(initial_values.keys()) + tuple(equations.keys()):
        inventory[var.variable].add(var.order)

    # TODO. it is also a parameter if it depends on time.

    # or RHS of the equations
    for group in (initial_values, equations):
        for value in group.values():
            if not hasattr(value, "yield_named"):
                continue
            for named in value.yield_named():
                if isinstance(named, Derivative):
                    inventory[named.variable].add(named.order)
                elif isinstance(named, Variable):
                    inventory[named].add(0)
                elif isinstance(named, (Constant, Parameter)):
                    # Constant are automatically added to parameters
                    parameters.add(named)

    for var, orders in inventory.items():
        if len(orders) == 1:
            parameters.add(var)
        else:
            variables.add(var)
            assert var.equation_order == max(orders)

    #############
    # Step 2
    # Replace Variable/Derivative/Constant by SimpleVariable and SimpleParameter
    # according to the previous categorization
    # and add first order equations

    mapper = SystemContentMapper(variables, parameters)

    # Initial values
    # Map variable to value.
    ivs: dict[SimpleVariable, Any] = {
        substitute(k, mapper): substitute(v, mapper) for k, v in initial_values.items()
    }

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

    for var in sorted(variables):
        # For each variable
        # - create first order differential equations except for var.equation_order
        # - for the var.equation_order use the defined equation
        assert var.equation_order is not None
        current: Variable | Derivative = var
        for _ in range(var.equation_order - 1):
            upcoming = current.derive()
            deqs[SimpleVariable(str(current))] = SimpleVariable(str(upcoming))
            current = upcoming

        lhs = SimpleVariable(str(current))
        rhs = substitute(equations[current.derive()], mapper)
        deqs[lhs] = rhs

    # The state variables are all the keys in the differential equations
    # TODO: add algebraic equations
    state_variables = tuple(deqs.keys())

    return (
        ivs,
        aeqs,
        deqs,
        state_variables,
        tuple(substitute(k, mapper) for k in parameters),
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
    ivs, aeqs, deqs, state_variables, parameters = build_first_order_symbolic_ode(
        system
    )

    state_names = tuple(sorted(str(v) for v in state_variables))
    param_names = tuple(sorted(str(p) for p in parameters))

    ivs = {str(k): ode_vectorize(v, state_names, param_names) for k, v in ivs.items()}
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

    initial_body = [
        assignment_func("y0", to_index(k), str(eq)) for k, eq in ivs.items()
    ]
    update_param_body = [
        assignment_func("p", to_index(k), str(eq)) for k, eq in aeqs.items()
    ]
    ode_step_body = [
        assignment_func("ydot", to_index(k), str(eq)) for k, eq in deqs.items()
    ]

    initial_def = "\n    ".join(
        [
            "def init(t, p, y0):",
            *initial_body,
            "return y0",
        ]
    )

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

    return Compiled(
        state_names,
        param_names,
        initial_def,
        ode_step_def,
        update_param_def,
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
                vectorized.init_func,
                vectorized.ode_func,
                vectorized.param_func,
            ]
        ),
        libsl,
    )

    return Compiled(
        vectorized.variable_names,
        vectorized.parameter_names,
        optimizer(lm["init"]),
        optimizer(lm["ode_step"]),
        optimizer(lm["update_param"]),
    )


@dataclass(frozen=True)
class Compiled(Generic[T]):
    variable_names: tuple[str, ...]
    parameter_names: tuple[str, ...]
    init_func: T
    ode_func: T
    param_func: T


def compile(
    system: System | type[System],
    backend: Backend = Backend.FIRST_ORDER_VECTORIZED_NUMPY_NUMBA,
) -> Compiled[RHS]:
    match backend:
        case Backend.FIRST_ORDER_VECTORIZED_STD:
            from symbolite.impl import libstd

            return build_first_order_functions(system, libstd)
        case Backend.FIRST_ORDER_VECTORIZED_NUMPY:
            from symbolite.impl import libnumpy

            return build_first_order_functions(system, libnumpy)
        case Backend.FIRST_ORDER_VECTORIZED_NUMPY_NUMBA:
            import numba
            from symbolite.impl import libnumpy

            return build_first_order_functions(system, libnumpy, numba.njit)
        case Backend.FIRST_ORDER_VECTORIZED_JAX:
            import jax
            from symbolite.impl import libjax

            return build_first_order_functions(
                system, libjax, jax.jit, assignment_func=jax_assignment
            )
        case _:
            assert_never(backend, message="Unknown backend {}")


def assert_never(arg: Never, *, message: str) -> Never:
    raise ValueError(message.format(arg))
