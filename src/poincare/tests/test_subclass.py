from pytest import raises

from ..types import System, Variable, initial


def test_derivative_outside_class():
    class Base(System):
        x: Variable = initial(default=0)

    Base.x.derive(initial=0)
    assert 1 not in Base.x.derivatives


def test_derivative_in_subclass():
    class Base(System):
        x: Variable = initial(default=0)

    with raises(TypeError):

        class Extended(Base):
            vx = Base.x.derive(initial=0)

    assert 1 not in Base.x.derivatives


def test_equation_outside_class():
    class Base(System):
        x: Variable = initial(default=0)

    Base.x.derive() << 1
    assert Base.x.equation_order is None


def test_equation_in_subclass():
    class Base(System):
        x: Variable = initial(default=0)

    with raises(TypeError):

        class Extended(Base):
            eq = Base.x.derive() << 1

    assert Base.x.equation_order is None
