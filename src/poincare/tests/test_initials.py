from pytest import mark

from .. import Derivative, System, Variable, initial


def test_variable():
    default = 0
    value = 1

    class Model(System):
        x: Variable = initial(default=default)

    assert Model.x.initial == default
    assert Model().x.initial == default
    assert Model(x=value).x.initial == value


def test_derivative():
    default = 0
    value = 1

    class Model(System):
        x: Variable = initial(default=0)
        vx: Derivative = x.derive(initial=default)

    assert Model.vx.initial == default
    assert Model().vx.initial == default
    assert Model(vx=value).vx.initial == value


@mark.parametrize(
    "values",
    [
        dict(x=1),
        dict(vx=1),
        dict(x=1, vx=1),
        dict(vx=1, x=1),
    ],
)
def test_variable_and_derivative(values):
    class Model(System):
        x: Variable = initial(default=0)
        vx: Derivative = x.derive(initial=0)

    defaults = dict(x=0, vx=0)
    model = Model()
    for k, v in defaults.items():
        assert getattr(Model, k).initial == v
        assert getattr(model, k).initial == v

    expected = {k: values.get(k, v) for k, v in defaults.items()}
    model = Model(**values)
    for k, v in expected.items():
        assert getattr(model, k).initial == v
