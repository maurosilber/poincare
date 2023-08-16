from pytest import raises
from symbolite import Symbol

from ...simulator import Simulator
from ...types import Constant, Derivative, Number, Parameter, System, Variable


class Model(System):
    k0 = Constant(default=0)
    k1 = Constant(default=k0)
    k2 = Constant(default=k1)
    k3 = Constant(default=k1 + k2)
    k4 = Constant(default=2 * k1)
    p = Parameter(default=k3 + 1)
    x = Variable(initial=k3 + k4)
    eq = x.derive() << p * x


def assert_initials(
    system: System | type[System],
    values: dict[Constant | Parameter | Variable | Derivative, Number | Symbol],
    *,
    expected_parameters: dict[Parameter, float],
    expected_variables: dict[Variable, float],
):
    sim = Simulator(system)
    problem = sim.create_problem(values)
    assert problem.p == expected_parameters
    assert problem.y == expected_variables


def test_default_initials():
    assert_initials(
        Model,
        values={},
        expected_parameters={Model.p: 1},
        expected_variables={Model.x: 0},
    )


def test_override_initials():
    assert_initials(
        Model,
        values={Model.k3: 1},
        expected_parameters={Model.p: 2},
        expected_variables={Model.x: 1},
    )
    assert_initials(
        Model,
        values={Model.k2: 1},
        expected_parameters={Model.p: 2},
        expected_variables={
            Model.x: 1,
        },
    )
    assert_initials(
        Model,
        values={Model.k1: 1},
        expected_parameters={Model.p: 3},
        expected_variables={
            Model.x: 4,
        },
    )
    assert_initials(
        Model,
        values={Model.k0: 1},
        expected_parameters={Model.p: 3},
        expected_variables={
            Model.x: 4,
        },
    )
    assert_initials(
        Model,
        values={Model.k0: 1, Model.k2: 2},
        expected_parameters={Model.p: 4},
        expected_variables={
            Model.x: 5,
        },
    )


def test_override_with_constants():
    assert_initials(
        Model,
        values={Model.k0: 2, Model.k1: 1, Model.k2: Model.k0},
        expected_parameters={Model.p: 4},
        expected_variables={Model.x: 5},
    )


def test_cyclic_initials():
    with raises(ValueError, match="Cyclic"):
        Simulator(Model).create_problem(values={Model.k0: Model.k2})
