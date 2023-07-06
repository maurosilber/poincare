from pytest import mark, raises

from .. import Derivative, System, Variable, initial


@mark.parametrize("default", [None, 0])
@mark.parametrize("value", [None, 1])
def test_variable(default, value):
    class Model(System):
        x: Variable = initial(default=default)

    assert Model.x.initial == default

    if default is None:
        with raises(TypeError):
            Model()
    else:
        assert Model().x.initial == default

    if value is None:
        with raises(TypeError):
            Model(x=value)
    else:
        assert Model(x=value).x.initial == value


@mark.parametrize("default", [None, 0])
@mark.parametrize("value", [None, 1])
def test_derivative(default, value):
    class Model(System):
        x: Variable = initial(default=0)
        vx: Derivative = x.derive(initial=default)

    assert Model.vx.initial == default

    if default is None:
        with raises(TypeError):
            Model()
    else:
        assert Model().vx.initial == default

    if value is None:
        with raises(TypeError):
            Model(vx=value)
    else:
        assert Model(vx=value).vx.initial == value


@mark.parametrize("default_var", [None, 0])
@mark.parametrize("value_var", [None, 1])
@mark.parametrize("default_der", [None, 0])
@mark.parametrize("value_der", [None, 1])
def test_variable_and_derivative(default_var, value_var, default_der, value_der):
    class Model(System):
        x: Variable = initial(default=default_var)
        vx: Derivative = x.derive(initial=default_der)

    defaults = dict(x=default_var, vx=default_der)
    values = dict(x=value_var, vx=value_der)

    for k, v in defaults.items():
        assert getattr(Model, k).initial == v

    if any(v is None for v in defaults.values()):
        with raises(TypeError):
            Model()
    elif all(v is not None for v in defaults.values()):
        model = Model()
        for k, v in defaults.items():
            assert getattr(model, k).initial == v

    if any(v is None for v in values.values()):
        with raises(TypeError):
            Model(**values)
    elif all(v is not None for v in values.values()):
        model = Model(**values)
        for k, v in values.items():
            assert getattr(model, k).initial == v

    values = {k: v for k, v in values.items() if v is not None}
    expected = {k: values.get(k, v) for k, v in defaults.items()}

    if any(v is None for v in expected.values()):
        with raises(TypeError):
            Model(**values)
    else:
        model = Model(**values)
        for k, v in expected.items():
            assert getattr(model, k).initial == v
