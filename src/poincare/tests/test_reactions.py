from ..compile import get_equations
from ..types import EquationGroup, System, Variable, initial
from .utils import is_same_variable


def test_first_order_equation():
    class Model(System):
        x: Variable = initial(default=0)
        y: Variable = initial(default=0)
        reaction = EquationGroup(
            x.derive() << x,
        )

    for m in [Model, Model()]:
        m: Model
        assert is_same_variable(m.x, m.reaction.equations[0].lhs.variable)
        equations = get_equations(m)
        assert len(equations) == 1
        assert equations[m.x.derive()] == [m.reaction.equations[0].rhs]
