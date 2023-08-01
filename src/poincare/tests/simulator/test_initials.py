from pytest import raises

from ... import Constant, Parameter, System, Variable
from ...simulator import Simulator


class Model(System):
    k0 = Constant(default=0)
    k1 = Constant(default=k0)
    k2 = Constant(default=k1)
    k3 = Constant(default=k1 + k2)
    k4 = Constant(default=2 * k1)
    p = Parameter(default=k3 + 1)
    x = Variable(initial=k3 + k4)


def test_default_initials():
    assert Simulator(Model)._resolve_initials() == {
        Model.k0: 0,
        Model.k1: 0,
        Model.k2: 0,
        Model.k3: 0,
        Model.k4: 0,
        Model.p: 1,
        Model.x: 0,
    }


def test_override_initials():
    assert Simulator(Model)._resolve_initials(values={Model.k3: 1}) == {
        Model.k0: 0,
        Model.k1: 0,
        Model.k2: 0,
        Model.k3: 1,
        Model.k4: 0,
        Model.p: 2,
        Model.x: 1,
    }
    assert Simulator(Model)._resolve_initials(values={Model.k2: 1}) == {
        Model.k0: 0,
        Model.k1: 0,
        Model.k2: 1,
        Model.k3: 1,
        Model.k4: 0,
        Model.p: 2,
        Model.x: 1,
    }
    assert Simulator(Model)._resolve_initials(values={Model.k1: 1}) == {
        Model.k0: 0,
        Model.k1: 1,
        Model.k2: 1,
        Model.k3: 2,
        Model.k4: 2,
        Model.p: 3,
        Model.x: 4,
    }
    assert Simulator(Model)._resolve_initials(values={Model.k0: 1}) == {
        Model.k0: 1,
        Model.k1: 1,
        Model.k2: 1,
        Model.k3: 2,
        Model.k4: 2,
        Model.p: 3,
        Model.x: 4,
    }
    assert Simulator(Model)._resolve_initials(values={Model.k0: 1, Model.k2: 2}) == {
        Model.k0: 1,
        Model.k1: 1,
        Model.k2: 2,
        Model.k3: 3,
        Model.k4: 2,
        Model.p: 4,
        Model.x: 5,
    }


def test_override_with_constants():
    assert Simulator(Model)._resolve_initials(
        values={Model.k0: 2, Model.k1: 1, Model.k2: Model.k0}
    ) == {
        Model.k0: 2,
        Model.k1: 1,
        Model.k2: 2,
        Model.k3: 3,
        Model.k4: 2,
        Model.p: 4,
        Model.x: 5,
    }


def test_cyclic_initials():
    with raises(RecursionError, match="cyclic"):
        Simulator(Model)._resolve_initials(values={Model.k0: Model.k2})
