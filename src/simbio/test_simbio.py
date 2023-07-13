from . import MassAction, Reaction, Species, System, Variable, initial


def test_one_reactant():
    class Model(System):
        x: Variable = initial(default=0)
        eq = Reaction(reactants=[x], rate_law=lambda x: 42 * x.variable)

    m = Model()
    assert m.eq.reactants == [Species(m.x, 1)]
    assert m.eq.products == []

    assert len(m.eq.equations) == 1
    assert m.eq.equations[0].lhs == m.x.derive()
    assert m.eq.equations[0].rhs == -1.0 * (42 * m.x)


def test_one_product():
    class Model(System):
        x: Variable = initial(default=0)
        eq = Reaction(products=[x], rate_law=lambda: 42)

    m = Model()
    assert m.eq.reactants == []
    assert m.eq.products == [Species(m.x, 1)]

    assert len(m.eq.equations) == 1
    assert m.eq.equations[0].lhs == m.x.derive()
    assert m.eq.equations[0].rhs == 1.0 * 42


def test_one_reactant_with_stoichiometry():
    class Model(System):
        x: Variable = initial(default=0)
        eq = Reaction(reactants=[2 * x], rate_law=lambda x: 42 * x.variable)

    m = Model()
    assert m.eq.reactants == [Species(m.x, 2)]
    assert m.eq.products == []

    assert len(m.eq.equations) == 1
    assert m.eq.equations[0].lhs == m.x.derive()
    assert m.eq.equations[0].rhs == -2.0 * (42 * m.x)


def test_one_product_with_stoichiometry():
    class Model(System):
        x: Variable = initial(default=0)
        eq = Reaction(products=[2 * x], rate_law=lambda: 42)

    m = Model()
    assert m.eq.reactants == []
    assert m.eq.products == [Species(m.x, 2)]

    assert len(m.eq.equations) == 1
    assert m.eq.equations[0].lhs == m.x.derive()
    assert m.eq.equations[0].rhs == 2.0 * 42


def test_same_reactant_and_product():
    class Model(System):
        x: Variable = initial(default=0)
        eq = Reaction(
            reactants=[2 * x],
            products=[3 * x],
            rate_law=lambda x: 42 * x.variable,
        )

    m = Model()
    assert m.eq.reactants == [Species(m.x, 2)]
    assert m.eq.products == [Species(m.x, 3)]

    assert len(m.eq.equations) == 1
    assert m.eq.equations[0].lhs == m.x.derive()
    assert m.eq.equations[0].rhs == 1.0 * (42 * m.x)


def test_same_reactant_and_product_with_mass_action():
    class Model(System):
        x: Variable = initial(default=0)
        eq = MassAction(
            reactants=[2 * x],
            products=[3 * x],
            rate=42,
        )

    m = Model()
    assert m.eq.reactants == [Species(m.x, 2)]
    assert m.eq.products == [Species(m.x, 3)]

    assert len(m.eq.equations) == 1
    assert m.eq.equations[0].lhs == m.x.derive()
    assert m.eq.equations[0].rhs == 1.0 * (42 * m.x**2)
    equations = list(m.eq.yield_equations())
    assert len(equations) == 1
    assert equations[0].lhs == m.x.derive()
    assert equations[0].rhs == 1.0 * (42 * m.x**2)
