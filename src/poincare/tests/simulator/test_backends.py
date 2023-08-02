import numpy as np
from pytest import mark

from ... import Parameter, System, Variable
from ...simulator import Backend, Simulator


class Oscillator(System):
    x = Variable(initial=1)
    vx = x.derive(initial=0)
    spring_constant = Parameter(default=1)
    spring = vx.derive() << -spring_constant * x


def solution(t, x=1):
    return x * np.cos(t)


@mark.parametrize("backend", Backend)
def test_solve(backend):
    t = np.linspace(0, 10, 100)

    sim = Simulator(Oscillator, backend=backend)
    result = sim.solve(times=t)
    assert np.allclose(result["x"], solution(t))
