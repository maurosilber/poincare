import matplotlib.pyplot as plt
import numpy as np
from poincare import Derivative, Parameter, System, Variable, assign, initial
from poincare.simulator import Simulator
from symbolite import scalar

t = System.time


class Oscillator(System):
    x: Variable = initial(default=1)
    vx: Derivative = x.derive(initial=0)
    phase: Parameter = assign(default=0)
    F: Parameter = assign(default=scalar.cos(t + phase))
    spring = vx.derive() << -x + F


if __name__ == "__main__":
    sim = Simulator(Oscillator)
    result = sim.solve(times=np.linspace(0, 50, 1000))
    result["x"].rename("cos(t)").plot()

    # No recompilation necessary to change parameter value
    result = sim.solve(times=np.linspace(0, 50, 1000), values={Oscillator.phase: -1.3})
    result["x"].rename("cos(t-1.3)").plot()

    # To change functional form, must be recompiled
    sim = Simulator(Oscillator(F=scalar.sin(t)))
    result = sim.solve(times=np.linspace(0, 50, 1000))
    result["x"].rename("sin(t)").plot()

    plt.legend()
    plt.show()
