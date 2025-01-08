from __future__ import annotations

from poincare import Derivative, Simulator, System, Variable, initial
from poincare._call import Function


def example(x0, x1):
    return x1, -x0


class Oscillator(System):
    x: Variable = initial(default=1)
    v: Derivative = x.derive(initial=0)

    eqs = [x, v] << Function(example).using(x, v)


sim = Simulator(Oscillator)
df = sim.solve(t_span=(0, 10))
df.plot()
