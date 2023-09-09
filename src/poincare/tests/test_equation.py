from pytest import mark, raises

from ..compile import get_equations
from ..types import Derivative, Parameter, System, Variable, assign, initial
from .utils import is_same_variable


def test_first_order_equation():
    class Model(System):
        x: Variable = initial(default=0)
        prop = x.derive() << x

    assert is_same_variable(Model.prop.lhs, Model.x.derive())
    assert Model.prop.rhs == Model.x

    model = Model(x=1)
    assert is_same_variable(model.prop.lhs, model.x.derive())
    assert model.prop.rhs == model.x

    for m in [Model, model]:
        equations = get_equations(m)
        x = m.x
        assert equations[Derivative(x, order=x.equation_order)] == [x]


def test_do_not_reassign_initial():
    class Model(System):
        x: Variable = initial(default=0)
        vx = x.derive(initial=0)
        x.derive()

    assert Model.vx.initial == 0


def test_error_assigning_to_derivative_with_initial():
    with raises(ValueError):

        class Model(System):
            x: Variable = initial(default=0)
            vx = x.derive(initial=0)
            prop = x.derive() << x


@mark.xfail(reason="not implemented")
def test_error_assigning_to_derivative_with_annotation():
    with raises(ValueError):

        class Model(System):
            x: Variable = initial(default=0)
            vx: Derivative = x.derive()
            prop = x.derive() << x


def test_second_order_equation():
    class Model(System):
        x: Variable = initial(default=0)
        vx = x.derive(initial=0)

        force = vx.derive() << -x

    assert is_same_variable(Model.force.lhs, Model.vx.derive())
    assert Model.force.rhs == -Model.x

    model = Model(x=1)
    assert is_same_variable(model.force.lhs, model.vx.derive())
    assert model.force.rhs == -model.x

    for m in [Model, model]:
        equations = get_equations(m)
        x = m.x
        assert equations[Derivative(x, order=x.equation_order)] == [-x]


def test_second_order_equation_with_derivative():
    class Model(System):
        x: Variable = initial(default=0)
        vx = x.derive(initial=0)

        force = vx.derive() << -vx

    assert is_same_variable(Model.force.lhs, Model.vx.derive())
    assert Model.force.rhs == -Model.vx

    model = Model(x=1)
    assert is_same_variable(model.force.lhs, model.vx.derive())
    assert model.force.rhs == -model.vx

    for m in [Model, model]:
        equations = get_equations(m)
        x = m.x
        assert equations[Derivative(x, order=x.equation_order)] == [-m.vx]


def test_second_order_equation_without_first_derivative():
    """Taking the second derivative 'directly',
    without defining the first derivative."""

    with raises(NameError, match="assign"):

        class Model(System):
            x: Variable = initial(default=0)
            force = x.derive(initial=0.0).derive() << -x.derive()


def test_repeated_equations():
    class Model(System):
        x: Variable = initial(default=0)
        eq1 = x.derive() << 1
        eq2 = x.derive() << x

    for m in [Model, Model(x=1)]:
        x: Variable = m.x
        assert x.equation_order == 1
        equations = get_equations(m)
        dx = Derivative(x, order=x.equation_order)
        assert len(equations[dx]) == 2
        assert equations[dx] == [1, x]


def test_two_variable_equations():
    class Model(System):
        x: Variable = initial(default=0)
        y: Variable = initial(default=0)
        eq_x = x.derive() << x + y
        eq_y = y.derive() << x + y

    for obj in (
        Model,
        Model(x=1),
        Model(y=1),
        Model(x=1, y=1),
        Model(y=1, x=1),
    ):
        equations = get_equations(obj)
        for variable in [obj.x, obj.y]:
            assert variable.equation_order == 1
            derivative = Derivative(variable, order=variable.equation_order)
            assert len(equations[derivative]) == 1
            assert equations[derivative] == [obj.x + obj.y]


def test_compose_equations():
    class Constant(System):
        x: Variable = initial(default=0)
        eq = x.derive() << 1

    class Proportional(System):
        x: Variable = initial(default=0)
        eq = x.derive() << x

    class Model(System):
        x: Variable = initial(default=0)
        const = Constant(x=x)
        prop = Proportional(x=x)

    for m in [Model, Model(x=1)]:
        equations = get_equations(m)

        x: Variable = m.x
        assert x.equation_order == 1
        dx = Derivative(x, order=x.equation_order)
        assert len(equations[dx]) == 2
        assert equations[dx] == [1, x]


def test_compose_equations_with_derivatives():
    class ConstantIncrease(System):
        x: Variable = initial(default=0)
        eq = x.derive() << 1

    class Drag(System):
        x: Variable = initial(default=0)
        vx: Derivative = x.derive(initial=0)
        eq = vx.derive() << vx

    with raises(ValueError, match="assigned"):

        class Model(System):
            x: Variable = initial(default=0)
            const = ConstantIncrease(x=x)
            prop = Drag(x=x)


def test_compose_multilevel_equations():
    class Level0(System):
        x: Variable = initial(default=0)
        eq = x.derive() << x

    class Level1(System):
        x: Variable = initial(default=0)
        eq = Level0(x=x)

    class Level2(System):
        x: Variable = initial(default=0)
        eq = Level1(x=x)

    for m in [Level2, Level2(x=1)]:
        equations = get_equations(m)

        x: Variable = m.x
        assert x.equation_order == 1

        dx = Derivative(x, order=x.equation_order)
        assert len(equations[dx]) == 1
        assert equations[dx] == [x]


def test_parameter_equation():
    class Model(System):
        t = Variable(initial=0)

        k: Parameter = assign(default=t)


@mark.xfail(reason="Not yet implemented")
def test_unassigned_equation():
    with raises(NameError):

        class UnassignedLeftShift(System):
            x = Variable(initial=0)
            x.derive() << 0


def test_shadowed_equations():
    with raises(NameError):

        class Model(System):
            x = Variable(initial=0)
            eq = x.derive() << 0
            eq = x.derive() << 1


def test_outer_composition():
    class Particle(System):
        x: Variable = initial(default=0)

    class Model(System):
        p: Particle = Particle(x=0)
        eq = p.x.derive() << p.x

    assert Model.p.x is Model.eq.lhs.variable
    assert Model.p.x is Model.eq.rhs

    model = Model()
    assert model.p.x is model.eq.lhs.variable
    assert model.p.x is model.eq.rhs

    model = Model(p=Particle(x=1))
    assert model.p.x is model.eq.lhs.variable
    assert model.p.x is model.eq.rhs
