import numpy as np
from pytest import mark

from ... import Constant, Parameter, System, Variable
from ...simulator import Simulator
from ...types import Independent


class Model(System):
    time = Independent(default=0)
    c = Constant(default=0)
    unused = Constant(default=0)
    x = Variable(initial=c)
    y = Variable(initial=0)
    k = Parameter(default=1)
    F = Parameter(default=time)

    eq_x = x.derive() << k * x
    eq_y = y.derive() << F


times = np.linspace(0, 10, 100)


def test_one_variable():
    df_all = Simulator(Model).solve(save_at=times)
    df = Simulator(Model, transform={"x": Model.x}).solve(save_at=times)

    assert len(df) == len(df_all)
    assert (df["x"] == df_all["x"]).all()
    assert set(df.columns) == {"x"}


def test_sum_variable():
    df_all = Simulator(Model).solve(save_at=times)
    df = Simulator(Model, transform={"sum": Model.x + Model.y}).solve(save_at=times)

    assert len(df) == len(df_all)
    assert (df["sum"] == df_all["x"] + df_all["y"]).all()
    assert set(df.columns) == {"sum"}


@mark.xfail(reason="not implemented")
def test_non_variable():
    # Should it shortcut and skip the solver?
    sim = Simulator(Model, transform={"c": Model.c})

    df = sim.solve(save_at=times)
    assert np.all(df["c"] == Model.c.default)

    df = sim.solve(save_at=times, values={Model.c: Model.c.default + 1})
    assert np.all(df["c"] == Model.c.default + 1)


def test_number():
    sim = Simulator(Model, transform={"my_number": 1})
    df = sim.solve(save_at=times)
    assert np.all(df["my_number"] == 1)


@mark.xfail(reason="not implemented")
def test_unused_variable():
    sim = Simulator(Model, transform={"unused": Model.unused})

    df = sim.solve(save_at=times)
    assert np.all(df["unused"] == Model.unused.default)

    df = sim.solve(save_at=times, values={Model.unused: Model.unused.default + 1})
    assert np.all(df["unused"] == Model.unused.default + 1)


@mark.xfail(reason="Optimization not implemented.")
def test_unused_variable_skips_solver():
    """If the transform does not need to integrate the equations, it could skip that."""
    sim = Simulator(Model, transform={"unused": Model.unused})
    df = sim.solve(save_at=times, solver=None)
    assert np.all(df["unused"] == Model.unused.default)
