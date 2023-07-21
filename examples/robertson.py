from poincare import Parameter, System, Variable, assign, initial


class Robertson(System):
    x1: Variable = initial(default=0)
    x2: Variable = initial(default=0)
    x3: Variable = initial(default=0)

    k1: Parameter = assign(default=0.04)
    k2: Parameter = assign(default=1e4)
    k3: Parameter = assign(default=3e7)

    dx1 = x1.derive() << -k1 * x1 + k2 * x2 * x3
    dx2 = x2.derive() << k1 * x1 - k2 * x2 * x3 - k3 * x2**2
    dx3 = x3.derive() << k3 * x2**2


class RobertsonConstrained(System):
    x1: Variable = initial(default=0)
    x2: Variable = initial(default=0)
    x3 = 1 - x1 - x2

    k1: Parameter = assign(default=0.04)
    k2: Parameter = assign(default=1e4)
    k3: Parameter = assign(default=3e7)

    dx1 = x1.derive() << -k1 * x1 + k2 * x2 * x3
    dx2 = x2.derive() << k1 * x1 - k2 * x2 * x3 - k3 * x2**2
