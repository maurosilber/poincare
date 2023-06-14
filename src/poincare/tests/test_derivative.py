from pytest import raises

from ..types import System, Variable
from .utils import is_derivative, is_same_variable


class Particle(System):
    x: Variable = Variable(default=0)
    vx: Variable = x.derive(default=1)


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


def test_explicit_assignment():
    class Model(System):
        x = Variable()
        vx = x.derive()
        p = Particle(x=x, vx=vx)

    assert is_same_variable(Model.p.x, Model.x)
    assert is_same_variable(Model.p.vx, Model.vx)

    # Direct relationship
    assert is_derivative(Model.vx, Model.x)

    # Maintain inner relationship from Particle
    assert is_derivative(Model.p.vx, Model.p.x)

    # The derivative is linked to the outside Variable
    assert is_derivative(Model.p.vx, Model.x)


def test_implicit_assignment():
    """Implicit assignment p.vx = vx

    The derivative is no explicitly passed as argument,
    but is created.
    """

    class Model(System):
        x = Variable()
        vx = x.derive()
        p = Particle(x=x)

    assert is_same_variable(Model.p.x, Model.x)

    # Maintain inner relationship from Particle
    assert is_derivative(Model.p.vx, Model.p.x)

    # The derivative is linked to the outside Variable
    assert is_derivative(Model.p.vx, Model.x)


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


def test_disallow_automatically_linked_integral():
    """Assigning only the derivarive is not allowed.
    The root variable of Particle must be explicitly assigned.

    This is the reverse behaviour from test_automatically_linked_variable.
    """

    with raises(TypeError, match="missing variable"):

        class WrongModel(System):
            vx = Variable()
            p = Particle(vx=vx)

    class Model(System):
        vx = Variable()
        p = Particle(x=vx.integral(), vx=vx)


def test_disallow_unlinked_variables():
    """Explicitly assigning variables which do not have
    a derivative relationship is not allowed.

    As the derivative relationship must be enforced inside Particle,
    it would create a hidden derivative relationship.
    """

    with raises(TypeError, match="unlinked"):

        class WrongModel(System):
            x = Variable()
            vx = Variable()
            p = Particle(x=x, vx=vx)
