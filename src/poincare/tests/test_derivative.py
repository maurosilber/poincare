from pytest import raises

from ..types import Derivative, System, Variable, initial
from .utils import is_derivative, is_same_variable


class Particle(System):
    x: Variable = initial(default=0)
    vx: Derivative = x.derive(initial=1)


def test_is_derivative():
    # As a class
    assert is_derivative(Particle.vx, Particle.x)

    # As an instance
    p = Particle()
    assert is_derivative(p.vx, p.x)


def test_duplicate_derivatives():
    """Taking the same derivative multiple times must create aliases,
    not new variables."""

    class Model(System):
        x = Variable()
        x1 = x.derive()
        x2 = x.derive()

    assert is_same_variable(Model.x1, Model.x2)
    assert is_derivative(Model.x1, Model.x)
    assert is_derivative(Model.x2, Model.x)


def test_automatic_derivative():
    """The derivative is not explicitly created,
    but is created to maintain the relationship inside Particle."""

    class Model(System):
        x = Variable()
        p = Particle(x=x)

    assert is_same_variable(Model.p.x, Model.x)

    # Maintain inner relationship from Particle
    assert is_derivative(Model.p.vx, Model.p.x)

    # The derivative is linked to the outside Variable
    assert is_derivative(Model.p.vx, Model.x)


def test_automatic_higher_order_derivative():
    """The derivative is not explicitly created,
    but is created to maintain the relationship inside Particle."""

    class SecondOrder(System):
        x0: Variable = initial()
        x1 = x0.derive()
        x2 = x1.derive()

    class Model(System):
        x = Variable()
        p = SecondOrder(x0=x)

    assert is_same_variable(Model.p.x0, Model.x)

    # Maintain inner relationship from Particle
    assert is_derivative(Model.p.x2, Model.p.x0)

    # The derivative is linked to the outside Variable
    assert is_derivative(Model.p.x2, Model.x)


def test_implicit_assignment():
    """Implicit assignment p.vx = vx"""

    class Model(System):
        x = Variable()
        vx: Derivative = x.derive()
        p = Particle(x=x)

    assert is_same_variable(Model.p.x, Model.x)

    # Maintain inner relationship from Particle
    assert is_derivative(Model.p.vx, Model.p.x)

    # The derivative is linked to the outside Variable
    assert is_derivative(Model.p.vx, Model.x)
    assert is_same_variable(Model.p.vx, Model.vx)


def test_raise_on_non_derivative():
    """Explicitly assigning variables which do not have
    a derivative relationship is not allowed.

    As the derivative relationship must be enforced inside Particle,
    it would create a hidden derivative relationship.
    """

    with raises(TypeError, match="initial"):

        class WrongModel(System):
            x = Variable()
            vx = Variable()
            p = Particle(
                x=x,
                vx=vx,  # type: ignore
            )


def test_raise_on_explicit_assignment():
    with raises(TypeError, match="initial"):

        class WrongModel(System):
            x = Variable()
            vx = x.derive()
            p = Particle(
                x=x,
                vx=vx,  # type: ignore
            )
