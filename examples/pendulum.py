from poincare import Constant, Derivative, System, Variable, assign, initial
from symbolite import scalar


class Pendulum(System):
    angle: Variable = initial(default=0)
    angular_velocity: Derivative = angle.derive(initial=0)

    pendulum_length: Constant = assign(default=0)
    gravity: Constant = assign(default=9.8)

    spring = angular_velocity.derive() << -gravity / pendulum_length * scalar.sin(angle)
