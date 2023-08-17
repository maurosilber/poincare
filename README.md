![Package](https://img.shields.io/pypi/v/poincare?label=poincare)
![CodeStyle](https://img.shields.io/badge/code%20style-black-000000.svg)
![License](https://img.shields.io/pypi/l/poincare?label=license)
![PyVersion](https://img.shields.io/pypi/pyversions/poincare?label=python)
[![CI](https://github.com/maurosilber/poincare/actions/workflows/ci.yml/badge.svg)](https://github.com/maurosilber/poincare/actions/workflows/ci.yml)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/maurosilber/poincare/main.svg)](https://results.pre-commit.ci/latest/github/maurosilber/poincare/main)

# Poincaré: simulation of dynamical systems

Poincaré allows to define and simulate dynamical systems in Python.

### Definition

To define the system

$$ \\frac{dx}{dt} = -x \\quad \\text{with} \\quad x(0) = 1 $$

we write can:

```python
>>> from poincare import Variable, System, initial
>>> class Model(System):
...   # Define a variable with name `x` with an initial value (t=0) of `1``.
...   x: Variable = initial(default=1)
...   # The rate of change of `x` (i.e. velocity) is assigned (<<) to `-x`.
...   # This relation is assigned to a Python variable (`eq`)
...   eq = x.derive() << -x
...
```

### Simulation

To simulate that system,
we do:

```python
>>> from poincare import Simulator
>>> sim = Simulator(Model)
>>> sim.solve(times=range(3))
             x
time
0     1.000000
1     0.367879
2     0.135335
```

The output is a `pandas.DataFrame`,
which can be plotted with `.plot()`.

### Changing initial conditions

To change the initial condition,
we have two options.

1. Passing a dictionary to the \`solve\`\` method:

```python
>>> sim.solve(values={Model.x: 2}, times=range(3))
             x
time
0     2.000000
1     0.735759
2     0.270671
```

which reuses the previously compiled model in the `Simulator` instance.

2. Instantiating the model with other values:

```python
>>> Simulator(Model(x=2)).solve(times=range(3))
             x
time
0     2.000000
1     0.735759
2     0.270671
```

This second option allows to compose systems
into bigger systems.
See the example in [examples/oscillators.py](https://github.com/maurosilber/poincare/blob/main/examples/oscillators.py).

### Transforming the output

We can compute transformations of the output
by passing a dictionary of expressions:

```python
>>> Simulator(Model, transform={"x": Model.x, "2x": 2 * Model.x}).solve(times=range(3))
             x        2x
time
0     1.000000  2.000000
1     0.367879  0.735759
2     0.135335  0.270671
```

### Higher-order systems

To define a higher-order system,
we have to assign an initial condition to the derivative of a variable:

```python
>>> from poincare import Derivative
>>> class Oscillator(System):
...   x: Variable = initial(default=1)
...   v: Derivative = x.derive(initial=0)
...   eq = v.derive() << -x
...
>>> Simulator(Oscillator).solve(times=range(3))
             x         v
time
0     1.000000  0.000000
1     0.540302 -0.841471
2    -0.416147 -0.909297
```

### Constants, Parameters, and functions

Besides variables,
we can define parameters and constants,
and use functions from [symbolite](https://github.com/hgrecco/symbolite).

#### Constants

Constants allow to define common initial conditions for Variables and Derivatives:

```python
>>> from poincare import assign, Constant
>>> class Model(System):
...     c: Constant = assign(default=1, constant=True)
...     x: Variable = initial(default=c)
...     y: Variable = initial(default=2 * c)
...     eq_x = x.derive() << -x
...     eq_y = y.derive() << -y
...
>>> Simulator(Model).solve(times=range(3))
             x         y
time
0     1.000000  2.000000
1     0.367879  0.735759
2     0.135335  0.270671
```

Now, we can vary their initial conditions jointly:

```python
>>> Simulator(Model(c=2)).solve(times=range(3))
             x         y
time
0     2.000000  4.000000
1     0.735759  1.471518
2     0.270671  0.541341
```

But we can break that connection by passing `y` initial value directly:

```python
>>> Simulator(Model(c=2, y=2)).solve(times=range(3))
             x         y
time
0     2.000000  2.000000
1     0.735759  0.735759
2     0.270671  0.270671
```

#### Parameters

Parameters are like Variables,
but their time evolution is given directly as a function of time,
Variables, Constants and other Parameters:

```python
>>> from poincare import Parameter
>>> class Model(System):
...     p: Parameter = assign(default=1)
...     x: Variable = initial(default=1)
...     eq = x.derive() << -p * x
...
>>> Simulator(Model).solve(times=range(3))
             x
time
0     1.000000
1     0.367879
2     0.135335
```

#### Functions

Symbolite functions are accessible from the `symbolite.scalar` module:

```python
>>> from symbolite import scalar
>>> class Model(System):
...     x: Variable = initial(default=1)
...     eq = x.derive() << scalar.sin(x)
...
>>> Simulator(Model).solve(times=range(3))
             x
time
0     1.000000
1     1.956295
2     2.655911
```

### Units

poincaré also supports functions through
[`pint`](https://github.com/hgrecco/pint)
and [`pint-pandas`](https://github.com/hgrecco/pint-pandas).

```python
>>> import pint
>>> unit = pint.get_application_registry()
>>> class Model(System):
...     x: Variable = initial(default=1 * unit.m)
...     v: Derivative = x.derive(initial=0 * unit.m/unit.s)
...     w: Parameter = assign(default=1 * unit.Hz)
...     eq = v.derive() << -w**2 * x
...
>>> result = Simulator(Model).solve(times=range(3))
```

The columns have units of m and m/s, respectively.
`pint` raises a `DimensionalityError` if we try to add them:

```python
>>> result["x"] + result["v"]
Traceback (most recent call last):
...
pint.errors.DimensionalityError: Cannot convert from 'meter' ([length]) to 'meter / second' ([length] / [time])
```

We can remove the units and set them as string metadata with:

```python
>>> result.pint.dequantify()
             x              v
unit     meter meter / second
time
0     1.000000       0.000000
1     0.540302      -0.841471
2    -0.416147      -0.909297
```

which allows to plot the DataFrame with `.plot()`.

## Installation

```bash
pip install -U poincare
```
