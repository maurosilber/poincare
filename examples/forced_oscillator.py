import matplotlib.pyplot as plt
import numpy as np
from poincare import Derivative, Parameter, System, Variable, assign, initial
from poincare.simulator import Simulator
from symbolite import scalar


class Oscillator(System):
    x: Variable = initial(default=1)
    vx: Derivative = x.derive(initial=0)
    spring_constant: Parameter = assign(default=1)
    F: Parameter = assign(default=scalar.cos(System.simulation_time))
    spring = vx.derive() << -spring_constant * x + F


if __name__ == "__main__":
    sim = Simulator(Oscillator)
    result = sim.solve(times=np.linspace(0, 50, 1000))
    result["x"].plot()
    plt.show()
