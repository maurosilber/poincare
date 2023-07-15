from pytest import mark, raises

from .. import Derivative, System, Variable, initial


@mark.xfail(reason="not implemented")
def test_variable():
    class Model(System):
        x: Variable = initial(default=0)

    with raises(TypeError):
        Model.x = 1

    with raises(TypeError):
        Model().x = 1


@mark.xfail(reason="not implemented")
def test_derivative():
    class Model(System):
        x: Variable = initial(default=0)
        vx: Derivative = x.derive(initial=0)

    with raises(TypeError):
        Model.vx = 1

    with raises(TypeError):
        Model().vx = 1
