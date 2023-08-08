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


def test_duplicate_derivative_assignment():
    """Taking the same derivative multiple times must create aliases,
    not new variables."""
    with raises(ValueError, match="assigned"):

        class WrongModel1(System):
            x = Variable(initial=0)
            x1 = x.derive(initial=0)
            x2 = x.derive(initial=1)

    with raises(ValueError, match="assigned"):

        class WrongModel2(System):
            x = Variable(initial=0)
            x1 = x.derive(initial=0)
            x2 = x.derive(initial=0)


def test_duplicate_derivatives():
    """Cannot take the same derivative multiple times."""

    with raises(NameError):

        class Model(System):
            x = Variable(initial=0)
            x1 = x.derive(initial=0)
            x2 = x.derive()


def test_no_automatic_derivative():
    """The derivative is not explicitly created,
    but is created to maintain the relationship inside Particle."""

    with raises(TypeError, match="derivative"):

        class Model(System):
            x = Variable(initial=0)
            p = Particle(x=x)


def test_explicit_assignment():
    class Model(System):
        x = Variable(initial=0)
        vx = x.derive(initial=0)
        p = Particle(x=x, vx=vx)

    assert is_same_variable(Model.p.x, Model.x)

    # Maintain inner relationship from Particle
    assert is_derivative(Model.p.vx, Model.p.x)

    # The derivative is linked to the outside Variable
    assert is_derivative(Model.p.vx, Model.x)
    assert is_same_variable(Model.p.vx, Model.vx)


def test_explicit_assignment_of_initial():
    with raises(TypeError):

        class Model1(System):
            x = Variable(initial=0)
            vx = x.derive(initial=0)
            p = Particle(vx=1, x=x)

    with raises(TypeError):

        class Model(System):
            x = Variable(initial=0)
            vx = x.derive(initial=0)
            p = Particle(x=x, vx=1)


def test_implicit_assignment():
    """Implicit assignment p.vx = vx"""

    class Model(System):
        x = Variable(initial=0)
        vx: Derivative = x.derive(initial=0)
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

    with raises(TypeError, match="derivative"):

        class WrongModel(System):
            x = Variable(initial=0)
            vx = Variable(initial=0)
            p = Particle(
                x=x,
                vx=vx,  # type: ignore
            )
