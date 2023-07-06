"""Test model instantiation."""

from pytest import raises

from ..types import Derivative, System, Variable, initial


def test_empty_system():
    class Model(System):
        pass

    Model()


def test_required_variable():
    """The variable is required to instantiate the system.

    Variables are keyword-only.
    """

    class Model(System):
        x: Variable = initial()

    with raises(TypeError, match="missing"):
        Model()  # type: ignore

    with raises(TypeError, match="positional"):
        Model(1)  # type: ignore

    Model(x=1)


def test_default_variable():
    class Model(System):
        x: Variable = initial(default=0)

    assert Model().x.initial == 0

    # Change instance value
    assert Model(x=1).x.initial == 1


def test_wrong_default_variable():
    with raises(TypeError):

        class Model(System):
            x: Variable = 0  # type: ignore


def test_derivative():
    class Model(System):
        x: Variable = initial(default=0)
        vx: Derivative = x.derive(initial=0)

    assert Model(vx=1).vx.initial == 1
    assert Model(x=1, vx=1).vx.initial == 1


def test_internal_variable():
    class Model(System):
        x = Variable(initial=0)

    Model()

    with raises(TypeError, match="unexpected"):
        Model(x=1)  # type: ignore


def test_extended_subclass():
    class Model(System):
        x: Variable = initial()

    class ExtendedModel(Model):
        y: Variable = initial()

    with raises(TypeError, match="missing"):
        ExtendedModel(x=1)  # type: ignore

    with raises(TypeError, match="missing"):
        ExtendedModel(y=1)  # type: ignore

    ExtendedModel(x=1, y=1)
