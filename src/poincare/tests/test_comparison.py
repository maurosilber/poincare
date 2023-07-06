"""Systems with the same type, components, and initial values are equal.

It allows hashing and using components in dictionaries.
"""

from ..types import System, Variable, initial


def test_initial_value():
    class Model(System):
        x: Variable = initial()

    assert Model(x=0) == Model(x=0)
    assert Model(x=0) != Model(x=1)


def test_default_initial_value():
    class Model(System):
        x: Variable = initial(default=0)

    assert Model() == Model(x=0)
    assert Model() != Model(x=1)


def test_empty_system():
    class Model1(System):
        pass

    class Model2(System):
        pass

    # Compare instances of the same
    assert Model1() == Model1()
    # Compare instances
    assert Model1() != Model2()


def test_subclass_equal():
    class Model(System):
        x: Variable = initial()

    class SubModel(Model):
        pass

    assert Model(x=1) != SubModel(x=1)


def test_subclass_extended():
    class Model(System):
        x: Variable = initial()

    class ExtendedModel(Model):
        y: Variable = initial()

    assert Model(x=1) != ExtendedModel(x=1, y=1)


def test_composition():
    class Subsystem(System):
        x: Variable = initial()

    class Model(System):
        s: Subsystem = Subsystem(x=0)

    assert Model() == Model(s=Subsystem(x=0))
    assert Model() != Model(s=Subsystem(x=1))