from poincare import Constant, Derivative, System, Variable, assign, initial


class Oscillator(System):
    x: Variable = initial(default=0)
    vx: Derivative = x.derive(initial=0)

    spring_constant: Constant = assign(default=0)

    spring = vx.derive() << -spring_constant * x


class Dampening(System):
    x: Variable = initial(default=0)
    vx: Derivative = x.derive(initial=0)

    damp_rate: Constant = assign(default=0)

    dampening = vx.derive() << -damp_rate * vx


class DampedOscilator(System):
    x: Variable = initial(default=0)
    vx: Derivative = x.derive(initial=0)

    spring_constant: Constant = assign(default=0)
    damp_rate: Constant = assign(default=0)

    osillator = Oscillator(x=x, spring_constant=spring_constant)
    dampening = Dampening(x=x, damp_rate=damp_rate)
