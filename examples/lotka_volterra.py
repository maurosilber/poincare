from poincare import Constant, System, Variable, assign, initial


class LoktaVolterra(System):
    x: Variable = initial(default=0)
    y: Variable = initial(default=0)

    a: Constant = assign(default=0)
    b: Constant = assign(default=0)
    c: Constant = assign(default=0)
    d: Constant = assign(default=0)

    birth_pray = x.derive(assign=a * x)
    death_pray = x.derive(assign=-b * x * y)

    birth_predator = y.derive(assign=d * x * y)
    death_predator = y.derive(assign=-c * y)
