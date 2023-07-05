from pytest import raises

from ..types import Constant, Scalar, System, Variable, assign, initial
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


def test_parameter_equation():
    class Model(System):
        t = Variable()

        k: Variable = assign(default=t)


def test_parameter_not_derivable():
    with raises(AttributeError):

        class Model(System):
            k = Constant(default=0)
            k.derive()
