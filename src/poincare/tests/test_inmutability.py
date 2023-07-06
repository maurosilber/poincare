from pytest import raises

from .. import Derivative, System, Variable, initial


def test_variable():
    class Model(System):
        x: Variable = initial(default=0)

    with raises(TypeError):
        Model.x = 1

    with raises(TypeError):
        Model().x = 1


def test_derivative():
    class Model(System):
        x: Variable = initial(default=0)
        vx: Derivative = x.derive(initial=0)

    with raises(TypeError):
        Model.vx = 1

    with raises(TypeError):
        Model().vx = 1
