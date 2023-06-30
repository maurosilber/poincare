from pytest import raises
from ..types import Constant, assign, System, Variable, initial, Call

from .utils import is_same_variable


def test_first_order_equation():
    class Model(System):
        x: Variable = initial(default=0)

        prop = x.derive() << x

    assert is_same_variable(Model.prop.lhs, Model.x.derive())
    assert Model.prop.rhs == Model.x


def test_second_order_equation():
    class Model(System):
        x: Variable = initial(default=0)
        vx = x.derive()

        force = vx.derive() << -x

    assert is_same_variable(Model.force.lhs, Model.vx.derive())
    assert Model.force.rhs == -Model.x


def test_second_order_equation_without_first_derivative():
    """Taking the second derivative 'directly',
    without defining the first derivative,
    does not allow to set its initial condition."""

    class Model(System):
        x: Variable = initial(default=0)

        prop = x.derive().derive() << x

    with raises(ValueError):
        Model()


def test_parameter_equation():
    class Model(System):
        t = Variable()

        k: Call = assign(default=t)


def test_parameter_not_derivable():
    with raises(AttributeError):
        class Model(System):
            k = Constant(default=0)
            k.derive()  # type: ignore


def test_parameter_not_ode_equation():
    with raises(TypeError, match="unsupported"):
        class Model(System):
            k = Constant(default=0)
            k << 0  # type: ignore
