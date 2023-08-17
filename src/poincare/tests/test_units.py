import numpy as np
from pint import DimensionalityError, get_application_registry
from poincare import (
    Constant,
    Derivative,
    Parameter,
    Simulator,
    System,
    Time,
    Variable,
    assign,
    initial,
)
from pytest import mark, raises
from symbolite import scalar
from symbolite.impl import libstd

u = get_application_registry()


@mark.parametrize(
    "value",
    [
        Time(default=1 * u.s),
        Constant(default=1 * u.s),
        Parameter(default=1 * u.s),
        Variable(initial=1 * u.s),
    ],
)
def test_symbol_and_quantity(value):
    q = 1 * u.s

    left = value + q
    right = q + value
    assert left.eval(libstd) == right.eval(libstd)

    left = left + q
    right = q + right
    assert left.eval(libstd) == right.eval(libstd)


def test_single_constant():
    class Model(System):
        x: Constant = assign(default=1 * u.m, constant=True)

    with raises(DimensionalityError):
        Model(x=1)
    with raises(DimensionalityError):
        Model(x=1 * u.m / u.s)


def test_single_parameter():
    class Model(System):
        x: Parameter = assign(default=1 * u.m)

    with raises(DimensionalityError):
        Model(x=1)
    with raises(DimensionalityError):
        Model(x=1 * u.m / u.s)


def test_single_variable():
    class Model(System):
        x: Variable = initial(default=1 * u.m)

    with raises(DimensionalityError):
        Model(x=1)
    with raises(DimensionalityError):
        Model(x=1 * u.m / u.s)


def test_single_derivative():
    class Model(System):
        x: Variable = initial(default=1 * u.m)
        v: Derivative = x.derive(initial=1 * u.m / u.s)

    with raises(DimensionalityError):
        Model(v=1)

    with raises(DimensionalityError):
        Model(v=1 * u.m)

    with raises(DimensionalityError):

        class NoUnits(System):
            x: Variable = initial(default=1 * u.m)
            v: Derivative = x.derive(initial=1)

    with raises(DimensionalityError):

        class WrongUnits(System):
            x: Variable = initial(default=1 * u.m)
            v: Derivative = x.derive(initial=1 * u.m)


def test_single_equation():
    class Model(System):
        x: Variable = initial(default=1 * u.m)
        eq = x.derive() << 1 * u.m / u.s

    with raises(DimensionalityError):

        class NoUnits(System):
            x: Variable = initial(default=1 * u.m)
            eq = x.derive() << 1

    with raises(DimensionalityError):

        class WrongUnits(System):
            x: Variable = initial(default=1 * u.m)
            eq = x.derive() << 1 * u.m

    with raises(DimensionalityError):

        class WrongVariableUnits(System):
            x: Variable = initial(default=1 * u.m)
            eq = x.derive() << x


def test_time():
    class Model(System):
        time = Time(default=0 * u.s)
        x: Variable = initial(default=1 * u.m)
        eq = x.derive() << x / (time + 1 * u.s)


@mark.parametrize(
    "func",
    [
        lambda x: Time(default=x),
        lambda x: Constant(default=x),
        lambda x: Parameter(default=x),
        lambda x: Variable(initial=x),
    ],
)
def test_dependencies(func):
    with raises(DimensionalityError):

        class WrongUnits(System):
            c0 = Constant(default=0 * u.s)
            c1 = func(c0 + 1)

    class Model(System):
        c0 = Constant(default=0 * u.s)
        c1 = func(c0 + 1 * u.s)


@mark.parametrize(
    "func",
    [
        lambda x: Time(default=x),
        lambda x: Constant(default=x),
        lambda x: Parameter(default=x),
        lambda x: Variable(initial=x),
    ],
)
def test_eval_required(func):
    class Model(System):
        x: Variable = initial(default=1 * u.m)
        y = func(None)
        eq = x.derive() << x * y
        # TODO: we don't know `y` units,
        # but we could infer what they must be,
        # and if they conflict somewhere else.


def test_function():
    with raises(DimensionalityError):

        class WrongUnits(System):
            time = Time(default=0 * u.s)
            p: Parameter = assign(default=scalar.cos(time))

    class Model(System):
        time = Time(default=0 * u.s)
        p: Parameter = assign(default=scalar.cos(time * u.Hz))


def test_normalization():
    class Model(System):
        T: Parameter = assign(default=1 * u.s)
        x: Variable = initial(default=1 * u.m)
        y: Variable = initial(default=1 * u.m)
        eq_x = x.derive() << (x - y) / T
        eq_y = y.derive() << (x + y) / T

    t = np.linspace(0, 1, 10)
    sim = Simulator(Model)
    df = sim.solve(times=t)
    df_cm = sim.solve(values={Model.y: 100 * u.cm}, times=t)
    assert np.allclose((df - df_cm).pint.dequantify().values, 0)
    assert df["y"].pint.units == u.m
    assert df_cm["y"].pint.units == u.cm


@mark.xfail(reason="Not yet implemented")
def test_unit_in_equation():
    class Model(System):
        x: Variable = initial(default=0 * u.m)
        eq = x.derive() << x / (1 * u.s)

    sim = Simulator(Model)
    sim.solve(times=np.linspace(0, 1, 10))


def test_zero_initial_with_unit():
    class Model(System):
        x: Variable = initial(default=0 * u.m)
        T: Parameter = assign(default=1 * u.s)
        eq = x.derive() << x / T

    times = np.linspace(0, 1, 10)
    sim = Simulator(Model)
    sim.solve(times=times)
