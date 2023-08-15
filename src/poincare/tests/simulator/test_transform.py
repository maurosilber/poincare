import numpy as np

from ... import Parameter, System, Variable
from ...simulator import Simulator
from ...types import Time


class Model(System):
    time = Time(default=0)
    x = Variable(initial=0)
    y = Variable(initial=0)
    k = Parameter(default=1)
    F = Parameter(default=time)

    eq_x = x.derive() << k * x
    eq_y = y.derive() << F


times = np.linspace(0, 10, 100)


def test_one_variable():
    df_all = Simulator(Model).solve(times=times)
    df = Simulator(Model, transform={"x": Model.x}).solve(times=times)

    assert len(df) == len(df_all)
    assert (df["x"] == df_all["x"]).all()
    assert set(df.columns) == {"x"}


def test_sum_variable():
    df_all = Simulator(Model).solve(times=times)
    df = Simulator(Model, transform={"sum": Model.x + Model.y}).solve(times=times)

    assert len(df) == len(df_all)
    assert (df["sum"] == df_all["x"] + df_all["y"]).all()
    assert set(df.columns) == {"sum"}
