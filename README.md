![Package](https://img.shields.io/pypi/v/poincare?label=poincare)
![CodeStyle](https://img.shields.io/badge/code%20style-black-000000.svg)
![License](https://img.shields.io/pypi/l/poincare?label=license)
![PyVersion](https://img.shields.io/pypi/pyversions/poincare?label=python)
[![CI](https://github.com/maurosilber/poincare/actions/workflows/test.yml/badge.svg)](https://github.com/maurosilber/poincare/actions/workflows/test.yml)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/maurosilber/poincare/main.svg)](https://results.pre-commit.ci/latest/github/maurosilber/poincare/main)

# poincaré: simulation of dynamical systems

Poincaré allows to define and simulate dynamical systems in Python

```python
>>> from poincare import Variable, System, initial
>>> class Model(System):
...   x = Variable(initial=1)
...   eq = x.derive() << -x
...
>>> from poincare.simulator import Simulator
>>> Simulator(Model).solve(times=range(5))
             x
time
0     1.000000
1     0.367879
2     0.135335
3     0.049787
4     0.018316
```

The output is a `pandas.DataFrame`,
which can be plotted with `.plot()`.

### Installation

```bash
pip install -U poincare
```
