import pickle

from pytest import mark

from .. import Derivative, System, Variable, initial


class EmptyModel(System):
    pass


class SingleVariable(System):
    x: Variable = initial(default=0)


class SingleDerivative(System):
    x: Variable = initial(default=0)
    v: Derivative = x.derive(initial=0)


class SingleEquation(System):
    x: Variable = initial(default=0)
    eq = x.derive() << -x


models = [
    EmptyModel,
    SingleVariable,
    SingleDerivative,
    SingleEquation,
]


@mark.parametrize("model", models)
def test_roundtrip(model: System):
    dump = pickle.dumps(model)
    load = pickle.loads(dump)
    assert load == model
