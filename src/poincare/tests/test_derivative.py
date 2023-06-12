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


def test_explicitly_linked_variable():
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


def test_automatically_linked_variable():
    """The derivative is not explicitly linked,
    but must maintain the relationship inside Particle."""

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

    This is the reverse behaviour from test_automatically_linked_variable.
    """

    with raises(TypeError, match="missing variable"):

        class WrongModel(System):
            vx = Variable()
            p = Particle(vx=vx)


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


def test_disallow_non_explicit_assignments():
    """Explicitly assigning only one variable
    would create an implicit assignment between
    derivatives or integrals."""

    with raises(TypeError, match="implicit assignment"):
        # Implicit assignment p.vx = vx
        class WrongModel(System):
            x = Variable()
            vx = x.derive()
            p = Particle(x=x)

    with raises(TypeError, match="implicit assignment"):
        # Implicit assignment p.x = x
        class WrongModel(System):
            x = Variable()
            vx = x.derive()
            p = Particle(vx=vx)


def test_explicit_assignments_all_the_way():
    """If we allow inner derivatives...
    
    Conflicts with:
        - test_allow_automatically_linked_variable (p.vx = vx)
        - test_disallow_non_explicit_assignments (p.x = x)
    """

    # Implicit derivative of x
    class InnerModel(System):
        x: Variable = Variable()
        p = Particle(x=x)

    class OuterModel(System):
        x = Variable()
        vx = x.derive()
        inner = InnerModel(x=x)  # implicit link inner.p.vx = vx

    assert is_same_variable(OuterModel.vx, OuterModel.inner.p.vx)
    assert is_derivative(OuterModel.inner.p.vx, OuterModel.x)
