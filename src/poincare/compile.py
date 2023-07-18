from collections import defaultdict
from typing import Generator
from types import ModuleType

from symbolite import Symbol, scalar, vector

from symbolite.core import evaluate, substitute
from .types import Equation, System, Variable, Initial, Derivative



def to_varname(var: Variable, order: int | None=0) -> str:
    if isinstance(var, Derivative):
        return to_varname(var.variable, var.order)
    return "root" + str(var) + f".{order}"


def get_equations(system: System | type[System]) -> dict[Variable, list[Equation]]:
    if isinstance(system, System):
        eqs = system.yield_equations()
    else:
        eqs = system.yield_equations(system)

    equations: dict[Variable, list[Equation]] = defaultdict(list)
    for eq in eqs:
        equations[eq.lhs.variable].append(eq)
    return equations


def get_first_order_ode(system: System) -> list[Equation]:
    equations = get_equations(system)
    return equations


def get_var_order_equations(system: System) -> Generator[tuple[tuple[Variable, int], Symbol], None, None]:
    for var, eqs in get_equations(system).items():
        order, eq = zip(*[(getattr(eq.lhs, "order", 0), eq.rhs) for eq in eqs])
        if len(set(order)) > 1:
            raise ValueError
        yield (var, order[0]), sum(eq) if len(eq) > 1 else eq[0]


class SelfEvalScalar(scalar.Scalar):
    """Special type of Scalar that is evaluated to itself.
    """

    def eval(self, libsl: ModuleType | None):
        return self
    

class ToODEScalarMapper:
    """Used to convert Poincare Variables into SelfEvalScalar.
    
    Benefetis:
    - leaner
    - immutable
    - understood by Symbolite to vectorize
    - can be involved when using evaluate to simplify part of the code.
    """

    @classmethod
    def get(self, key, default=None):
        if isinstance(key, Variable):
            return SelfEvalScalar(to_varname(key))
        return key
    



def build_ode(system: System, libsl: ModuleType):

    # Get all variable an equations in the subsytem (recursively)
    variables = tuple(system.yield_variables())

    # Build a list of variables 
    initial_values: dict[str, Initial] = {}

    # Maps derived variable to a Symbolite expression or constant
    # dx/dt = 4 maps {"x": 4}
    equations: dict[str, Initial | symbol.Symbol] = {}
    
    # Throught all this code `to_varname` will be used to 
    # obtain a unique name for a variable/derivative.

    for var in variables:

        # Store initial values in a dict
        # evaluating an expression (if present) and expressing everything
        # in terms of Scalars
        # might be a good moment to subsitute constants!
        for order, value in var.derivatives.items():
            initial_values[to_varname(var, order)] = evaluate(substitute(value, ToODEScalarMapper), libsl)

        # TODO: Es posible que un systema bien formado tenga esto igual a None? Que significa?
        # Es una ecuacion mal formada?
        if var.equation_order is not None:

            # Creates first order ODE extra equations. In the future this might be
            # done elsewhere and using Poincare Variables.
            for order in range(1, var.equation_order):
                equations[to_varname(var, order - 1)] = scalar.Scalar(to_varname(var, order))

    var_names = sorted(tuple(initial_values.keys()))

    eqs = {
        to_varname(var, order): vector.vectorize(
            evaluate(substitute(eq, ToODEScalarMapper), libsl), 
            var_names
        )
        for (var, order), eq in get_var_order_equations(system)
    }
    
    return eqs, initial_values
