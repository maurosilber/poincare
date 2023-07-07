from pytest import raises

from ..types import Constant, Derivative, System, Variable, assign, initial
from .utils import is_same_variable


def test_first_order_equation():
    class Model(System):
        x: Variable = initial(default=0)
        prop = x.derive(assign=x)

    assert is_same_variable(Model.prop.lhs, Model.x.derive())
    assert Model.prop.rhs == Model.x


def test_second_order_equation():
    class Model(System):
        x: Variable = initial(default=0)
        vx = x.derive(initial=0)

        force = vx.derive(assign=-x)

    assert is_same_variable(Model.force.lhs, Model.vx.derive())
    assert Model.force.rhs == -Model.x


def test_second_order_equation_without_first_derivative():
    """Taking the second derivative 'directly',
    without defining the first derivative."""

    class Model(System):
        x: Variable = initial(default=0)
        force = x.derive(initial=0.0).derive(assign=-x)


def test_repeated_equations():
    class Model(System):
        x: Variable = initial(default=0)
        eq1 = x.derive(assign=1)
        eq2 = x.derive(assign=x)

    assert Model.x.equation_order == 1
    assert len(Model.x.equations) == 2


def test_compose_equations():
    class Constant(System):
        x: Variable = initial(default=0)
        eq = x.derive(assign=1)

    class Proportional(System):
        x: Variable = initial(default=0)
        eq = x.derive(assign=x)

    class Model(System):
        x: Variable = initial(default=0)
        const = Constant(x=x)
        prop = Proportional(x=x)

    assert Model.x.equation_order == 1
    assert len(Model.x.equations) == 2


def test_compose_equations_with_derivatives():
    class ConstantIncrease(System):
        x: Variable = initial(default=0)
        eq = x.derive(assign=1)

    class Drag(System):
        x: Variable = initial(default=0)
        vx: Derivative = x.derive(initial=0)
        eq = vx.derive(assign=vx)

    with raises(ValueError, match="assigned"):

        class Model(System):
            x: Variable = initial(default=0)
            const = ConstantIncrease(x=x)
            prop = Drag(x=x)


def test_parameter_equation():
    class Model(System):
        t = Variable(initial=0)

        k: Variable = assign(default=t)


def test_parameter_not_derivable():
    with raises(AttributeError):

        class Model(System):
            k = Constant(default=0)
            k.derive()
