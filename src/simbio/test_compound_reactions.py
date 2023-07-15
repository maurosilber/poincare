from poincare.compile import get_equations

from . import Constant, MassAction, Species, System, Variable, assign, initial


def test_synthesis():
    class Synthesis(System):
        A: Variable = initial(default=0)
        B: Variable = initial(default=0)
        AB: Variable = initial(default=0)

        forward_rate: Constant = assign(default=0)
        reverse_rate: Constant = assign(default=0)

        forward_reaction = MassAction(
            reactants=[A, B],
            products=[AB],
            rate=forward_rate,
        )

        reverse_reaction = MassAction(
            reactants=[AB],
            products=[A, B],
            rate=reverse_rate,
        )

    class Model(System):
        x: Variable = initial(default=0)
        eq = Synthesis(A=x, forward_rate=1, reverse_rate=2)

    m = Model()
    assert m.eq.forward_reaction.reactants == [Species(m.x, 1), Species(m.eq.B, 1)]
    assert m.eq.forward_reaction.products == [Species(m.eq.AB, 1)]

    equations = get_equations(m)
    assert len(equations) == 3

    forward_rate = m.eq.forward_rate * m.x**1 * m.eq.B**1
    reverse_rate = m.eq.reverse_rate * m.eq.AB**1

    for var, st in {m.x: -1, m.eq.B: -1, m.eq.AB: 1}.items():
        set_of_rhs = set(eq.rhs for eq in equations[var])
        assert set_of_rhs == {st * forward_rate, (-st) * reverse_rate}
