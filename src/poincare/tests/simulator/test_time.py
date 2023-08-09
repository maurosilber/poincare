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

    with raises(ValueError, match="recompile"):
        sim.create_problem(values={Model.p: t})

    Simulator(Model(p=t))


def test_time_dependent_parameters():
    class Model(System):
        p: Parameter = assign(default=t)
        x: Variable = initial(default=0)
        eq = x.derive() << p

    sim = Simulator(Model)
    assert len(sim.compiled.parameters) == 0
    assert set(sim.compiled.param_funcs) == {Model.p}

    with raises(ValueError, match="recompile"):
        sim.create_problem(values={Model.p: 1})

    with raises(ValueError, match="recompile"):
        sim.create_problem(values={Model.p: t})

    Simulator(Model(p=t))


def test_variable_dependent_parameters():
    class Model(System):
        x: Variable = initial(default=0)
        eq = x.derive() << 1  # analogous to "time"
        p: Parameter = assign(default=x)
        y: Variable = initial(default=0)
        eq2 = y.derive() << p

    sim = Simulator(Model)
    assert len(sim.compiled.parameters) == 0
    assert set(sim.compiled.param_funcs) == {Model.p}

    with raises(ValueError, match="recompile"):
        sim.create_problem(values={Model.p: 1})

    with raises(ValueError, match="recompile"):
        sim.create_problem(values={Model.p: t})

    Simulator(Model(p=t))


def test_parameter_dependent_parameters():
    class Model(System):
        p0: Parameter = assign(default=0)
        p: Parameter = assign(default=p0)
        x: Variable = initial(default=0)
        eq = x.derive() << p

    sim = Simulator(Model)
    # only p is part of the vector of parameters
    assert set(sim.compiled.parameters) == {Model.p}
    assert len(sim.compiled.param_funcs) == 0
    # but initial values can be modified through p0
    assert sim.create_problem().p[0] == 0
    assert sim.create_problem(values={Model.p: 1}).p[0] == 1
    assert sim.create_problem(values={Model.p0: 1}).p[0] == 1

    # must recompile to assign a function to p or p0
    with raises(ValueError, match="recompile"):
        sim.create_problem(values={Model.p: t})

    with raises(ValueError, match="recompile"):
        sim.create_problem(values={Model.p0: t})

    # recompilation moves from parameter vector to parameter func
    model = Model(p=t)
    sim = Simulator(model)
    assert len(sim.compiled.parameters) == 0
    assert set(sim.compiled.param_funcs) == {model.p}

    # recompilation moves from parameter vector to parameter func
    model = Model(p0=t)
    sim = Simulator(model)
    assert len(sim.compiled.parameters) == 0
    assert set(sim.compiled.param_funcs) == {model.p, model.p0}
