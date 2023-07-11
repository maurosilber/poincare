"""Test model instantiation."""

from pytest import raises

from ..types import Derivative, System, Variable, initial


def test_empty_system():
    class Model(System):
        pass

    Model()


def test_error_on_positional():
    """System parameters are keyword-only."""

    class Model(System):
        x: Variable = initial(default=0)

    with raises(TypeError, match="positional"):
        Model(1)  # type: ignore


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
        x: Variable = initial(default=0)

    class ExtendedModel(Model):
        y: Variable = initial(default=0)

    ExtendedModel(x=1, y=1)
