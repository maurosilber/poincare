from pytest import raises

from ...simulator import Simulator
from ...types import Constant, Parameter, System, Variable, assign, initial

t = System.simulation_time


def test_no_time_dependent_parameters():
    class Model(System):
        c0: Constant = assign(default=0, constant=True)
        c1: Constant = assign(default=c0, constant=True)
        p: Parameter = assign(default=c1)
        x: Variable = initial(default=0)
        eq = x.derive() << p

    sim = Simulator(Model)
    assert set(sim.compiled.parameters) == {Model.p}
    assert len(sim.compiled.param_funcs) == 0

    sim.create_problem(values={Model.p: 1})

    with raises(ValueError):
        sim.create_problem(values={Model.p: t})

    Simulator(Model(p=t))
