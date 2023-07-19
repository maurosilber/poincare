from collections import defaultdict
from typing import Generator, Any, TypeAlias
from types import ModuleType
from dataclasses import dataclass
from functools import cache
from symbolite import Symbol, scalar, vector

from symbolite.core import evaluate, substitute, inspect

from .types import Equation, System, Variable, Initial, Derivative, Constant

def debug_print(d):
    print(f"--- {len(d)} elements ---")
    for k, v in d.items():
        print("key", str(k))
        print("value", str(v))
    print("-----------")


RHS: TypeAlias = Initial | Variable

def to_varname(var: Variable | Derivative, order: int | None=0) -> str:
    if isinstance(var, Derivative):
        return to_varname(var.variable, var.order)
    return str(var) + f".{order}"


def get_equations(system: System | type[System]) -> dict[Derivative, list[RHS]]:
    if isinstance(system, System):
        eqs = system.yield_equations()
    else:
        eqs = system.yield_equations(system)

    equations: dict[Derivative, list[RHS]] = defaultdict(list)
    for eq in eqs:
        equations[eq.lhs].append(eq.rhs)
    return equations


def get_initial_values(system: System | type[System]) -> dict[Derivative, Initial]:
    if isinstance(system, System):
        vars = system.yield_variables()
    else:
        vars = system.yield_variables(system)

    initial_values: dict[Derivative,  Initial] = {}
    for var in vars:
        for order, initial_value in var.derivatives.items():
            initial_values[Derivative(var, order=order)] = initial_value
    return initial_values


@dataclass(frozen=True)
class SimpleVariable(scalar.Scalar):
    """Special type of Scalar that is evaluated to itself.
    """

    def eval(self, libsl: ModuleType | None=None):
        return self

@dataclass(frozen=True)
class SimpleParameter(scalar.Scalar):
    """Special type of Scalar that is evaluated to itself.
    """

    def eval(self, libsl: ModuleType | None=None):
        return self
    

class ToSimpleScalar(dict[Any, Any]):
    """Used to convert Poincare Variables into SelfEvalScalar.
    
    Benefetis:
    - leaner
    - immutable
    - understood by Symbolite to vectorize
    - can be involved when using evaluate to simplify part of the code.
    """

    def __init__(self, vars, pars):
        self.vars = vars
        self.pars = pars

    def get(self, key: Any, default=None):
        if isinstance(key, Constant):
            return SimpleParameter(to_varname(key))
        elif isinstance(key, (Derivative, Variable)):
            if not isinstance(key, Derivative):
                key = Derivative(key, order=0)
            if key in self.pars:
                return SimpleParameter(to_varname(key))
            elif key in self.vars:
                return SimpleVariable(to_varname(key))
            else:
                raise ValueError(f"Not found {key}")
        return key


def eqsum(eqs: list[RHS]):
    if len(eqs) == 0:
        return 0
    elif len(eqs) == 1:
        return eqs[0]
    else:
        return sum(eqs)


def build_first_order_symbolic_ode(system: System,) -> tuple[
    dict[SimpleVariable, scalar.NumberT | Symbol], 
    dict[SimpleVariable, scalar.NumberT | Symbol],
    dict[SimpleVariable, scalar.NumberT | Symbol],
    tuple[SimpleVariable],
    tuple[SimpleParameter],
    ]:

    #############
    # Step 0:
    # Get initial value and flattened equations

    initial_values = get_initial_values(system)
    equations = {
        k: eqsum(v) 
        for k, v in get_equations(system).items()
        }
    
    debug_print(equations)

    #############
    # Step 1:
    # Divide between:
    # - parameter vector: variables that appear with in a single "order" and constants.
    # - state vector: variables appear with more than one "order".

    derivatives: set[Derivative] = set()
    param: set[Constant | Derivative] = set()

    # Create an inventory of all variables in the systems
    # taking note in which their derivative
    # Either in the LHS of equations and initial values
    inventory: defaultdict[Variable, set[int]] = defaultdict(set)
    for der in tuple(initial_values.keys()) + tuple(equations.keys()):
        inventory[der.variable].add(der.order)

    # TODO. it is also a parameter if it depends on time.

    # or RHS of the equations
    for group in (initial_values, equations):
        for value in group.values():
            if not hasattr(value, "yield_named"):
                continue
            for named in value.yield_named():
                if isinstance(named, Constant):
                    # Constant are automatically added to parameters
                    param.add(named)
                elif isinstance(named, Variable):
                    inventory[named].add(0)
                elif isinstance(named, Derivative):
                    inventory[named.variable].add(named.order)
    
    for var, orders in inventory.items():
        if len(orders) == 1:
            param.add(Derivative(var, order=orders.pop()))
        else:
            # also take not of the maximum order found!
            for order in orders:
                derivatives.add(Derivative(var, order=order))
            assert var.equation_order == max(orders)
        

    #############
    # Step 2
    # Replace Variable/Derivative/Constant by SimpleVariable and SimpleParameter
    # according to the previous categorization
    # and add first order equations

    mapper = ToSimpleScalar(derivatives, param)

    # Initial values 
    # Map variable to value.
    ivs: dict[SimpleVariable, Any] = {substitute(k, mapper): substitute(v, mapper) 
                                      for k, v in initial_values.items()}
    
    # Algebraic equations
    # Maps variable to equation.
    aeqs: dict[SimpleVariable, Any] = {substitute(k, mapper): substitute(v, mapper) 
                                       for k, v in equations.items()
                                       if k in param}

    # Differential equations
    # Map variable to be derived 1 time to equation.
    # (unlike 'equations' that maps derived variable to equation)
    deqs: dict[SimpleVariable, Any] = {}

    state: list[SimpleVariable] = []
    for der in derivatives:
        assert der.variable.equation_order is not None

        # All derivatives go into the state vecotr
        if der.order != der.variable.equation_order:
            deqs[SimpleVariable(to_varname(der))] = SimpleVariable(to_varname(der.derive()))
            state.append(SimpleVariable(to_varname(der)))
    
    seen = set()
    for der in derivatives:
        if der.variable in seen:
            continue
        assert der.variable.equation_order is not None
        deqs[SimpleVariable(to_varname(der.variable, der.variable.equation_order - 1))] = substitute(equations[Derivative(der.variable, order=der.variable.equation_order)], mapper) 
        seen.add(der.variable)

    debug_print(deqs)

    return (
        ivs, 
        aeqs,
        deqs, 
        tuple(state),
        tuple(substitute(k, mapper) for k in param), 
    )


def ode_vectorize(expr: scalar.NumberT | Symbol, state_names: tuple[str], param_names: tuple[str], state_varname: str="y", param_varname: str="p") -> scalar.NumberT | Symbol:
    expr = vector.vectorize(expr, state_names, varname=state_varname, scalar_type=SimpleVariable)
    expr = vector.vectorize(expr, param_names, varname=param_varname, scalar_type=SimpleParameter)
    return expr


def build_vectorized_first_order(system: System):
    initial_values, aeqs, differential_equations, state, param = build_first_order_symbolic_ode(system)
   
    state_names = tuple(sorted(str(v) for v in state))
    param_names = tuple(sorted(str(p) for p in param))

    # initial_values = {
    #     k: evaluate(v, libnumpy)
    #     for k, v in initial_values.items()
    # }

    # differential_equations = {
    #     k: evaluate(v, libnumpy)
    #     for k, v in differential_equations.items()
    # }

    initial_values = {
        str(k): ode_vectorize(v, state_names, param_names)
        for k, v in initial_values.items()
    }

    differential_equations = {
        str(k): ode_vectorize(v, state_names, param_names)
        for k, v in differential_equations.items()
    }

    def slhs(k: str, name: str) -> str:
        if k in state_names:
            return f"{name}[{state_names.index(k)}]"
        if k in param_names:
            return f"p[{param_names.index(k)}]"
        return f"Unknown variable <{k}>"
        raise ValueError(k)

    tab = " " * 4
    
    initial_body = "\n".join(f"{tab}{slhs(k, 'y0')} = {str(eq)}" for k, eq in initial_values.items())
    initial_def = f"def init(t, y, p, y0):\n{initial_body}\n{tab}return y0"""

    ode_step_body = "\n".join(f"{tab}{slhs(k, 'dy_dt')} = {str(eq)}" for k, eq in differential_equations.items())
    ode_step_def = f"def ode_step(t, y, p, dy):\n{ode_step_body}\n{tab}return dy_dt"""

    alg_step_body = "# nothing to see yet"
    alg_step_def = f"def alg_step(t, y, p, dy):\n{tab}{alg_step_body}\n{tab}return y"""


    return state_names, param_names, initial_def, ode_step_def, alg_step_def
