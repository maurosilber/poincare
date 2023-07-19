from pytest import mark, raises

from ..compile import get_equations
from ..types import Derivative, System, Variable, assign, initial
from .utils import is_same_variable


def test_first_order_equation():
    class Model(System):
        x: Variable = initial(default=0)
        prop = x.derive(assign=x)

    assert is_same_variable(Model.prop.lhs, Model.x.derive())
    assert Model.prop.rhs == Model.x
    equations = get_equations(Model)
    assert equations[Model.x][0].rhs == Model.x

    model = Model(x=1)
    assert is_same_variable(model.prop.lhs, model.x.derive())
    assert model.prop.rhs == model.x
    equations = get_equations(model)
    assert equations[model.x][0].rhs == model.x


def test_second_order_equation():
    class Model(System):
        x: Variable = initial(default=0)
        vx = x.derive(initial=0)

        force = vx.derive(assign=-x)

    assert is_same_variable(Model.force.lhs, Model.vx.derive())
    assert Model.force.rhs == -Model.x
    equations = get_equations(Model)
    assert equations[Model.x][0].rhs == -Model.x

    model = Model(x=1)
    assert is_same_variable(model.force.lhs, model.vx.derive())
    assert model.force.rhs == -model.x
    equations = get_equations(model)
    assert equations[model.x][0].rhs == -model.x


def test_second_order_equation_without_first_derivative():
    """Taking the second derivative 'directly',
    without defining the first derivative."""

    class Model(System):
        x: Variable = initial(default=0)
        force = x.derive(initial=0.0).derive(assign=-x)

    assert is_same_variable(Model.force.lhs, Model.x.derive().derive())
    assert Model.force.rhs == -Model.x
    equations = get_equations(Model)
    assert equations[Model.x][0].rhs == -Model.x

    model = Model(x=1)
    assert is_same_variable(model.force.lhs, model.x.derive().derive())
    assert model.force.rhs == -model.x
    equations = get_equations(model)
    assert equations[model.x][0].rhs == -model.x


def test_repeated_equations():
    class Model(System):
        x: Variable = initial(default=0)
        eq1 = x.derive(assign=1)
        eq2 = x.derive(assign=x)

    assert Model.x.equation_order == 1
    equations = get_equations(Model)
    assert len(equations[Model.x]) == 2
    assert equations[Model.x][0].rhs == 1
    assert equations[Model.x][1].rhs == Model.x

    model = Model(x=1)
    assert model.x.equation_order == 1
    equations = get_equations(model)
    assert len(equations[model.x]) == 2
    assert equations[model.x][0].rhs == 1
    assert equations[model.x][1].rhs == model.x


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
        for name in ["x", "y"]:
            variable: Variable = getattr(obj, name)
            assert variable.equation_order == 1
            equations = get_equations(obj)
            assert len(equations[variable]) == 1
            assert equations[variable][0].rhs == obj.x + obj.y


def test_compose_equations():
    class Constant(System):
        x: Variable = initial(default=0)
        eq = x.derive(assign=1)

    class Proportional(System):
        x: Variable = initial(default=0)
        eq = x.derive(assign=x)

    class Model(System):
        x: Variable = initial(default=0)
        const = Constant(x=x)
        prop = Proportional(x=x)

    assert Model.x.equation_order == 1
    equations = get_equations(Model)
    assert len(equations[Model.x]) == 2
    assert equations[Model.x][0].rhs == 1
    assert equations[Model.x][1].rhs == Model.x

    model = Model(x=1)
    assert model.x.equation_order == 1
    equations = get_equations(model)
    assert len(equations[model.x]) == 2
    assert equations[model.x][0].rhs == 1
    assert equations[model.x][1].rhs == model.x


def test_compose_equations_with_derivatives():
    class ConstantIncrease(System):
        x: Variable = initial(default=0)
        eq = x.derive(assign=1)

    class Drag(System):
        x: Variable = initial(default=0)
        vx: Derivative = x.derive(initial=0)
        eq = vx.derive(assign=vx)

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

    assert Level2.x.equation_order == 1
    equations = get_equations(Level2)
    assert len(equations[Level2.x]) == 1
    assert equations[Level2.x][0].rhs == Level2.x

    level2 = Level2(x=1)
    assert level2.x.equation_order == 1
    equations = get_equations(level2)
    assert len(equations[level2.x]) == 1
    assert equations[level2.x][0].rhs == level2.x


def test_parameter_equation():
    class Model(System):
        t = Variable(initial=0)

        k: Variable = assign(default=t)


def test_unassigned_equation():
    with raises(NameError):

        class UnassignedLeftShift(System):
            x = Variable(initial=0)
            x.derive() << 0

    with raises(NameError):

        class UnassignedDerive(System):
            x = Variable(initial=0)
            x.derive(assign=0)


def test_shadowed_equations():
    with raises(NameError):

        class Model(System):
            x = Variable(initial=0)
            eq = x.derive(assign=0)
            eq = x.derive(assign=1)


@mark.xfail
def test_outer_composition():
    class Particle(System):
        x: Variable = initial(default=0)

    class Model(System):
        p: Particle = Particle(x=0)
        eq = p.x.derive() << 1

    assert Model.p.x.equation_order == 1
    equations = get_equations(Model)
    assert len(equations[Model.p.x]) == 1
    assert equations[Model.p.x][0].rhs == 1

    m = Model()
    assert m.p.x.equation_order == 1
    equations = get_equations(m)
    assert len(equations[m.p.x]) == 1
    assert equations[m.p.x][0].rhs == 1
