from pytest import raises

from ..types import System, Variable
from .utils import is_derivative, is_same_variable


def test_system_comparison():
    """Models with the same components must be equal.

    It allows hashing and using components in dictionaries.
    """

    class Model1(System):
        pass

    class Model2(System):
        pass

    # Compare classes
    assert Model1 == Model2
    # Compare instances
    assert Model1() == Model2()
    # Compare instances of the same
    assert Model1() == Model1()


def test_required_variable():
    """The variable is required to instantiate the system.

    Variables are keyword-only.
    """

    class Model(System):
        x: Variable = Variable()

    with raises(TypeError, match="missing"):
        Model()

    with raises(TypeError, match="positional"):
        Model(1)

    Model(x=1)


def test_default_variable():
    class Model(System):
        x: Variable = Variable(default=1)

    # Compare classes
    assert Model == Model
    # Compare instances
    assert Model() == Model()
    # Change instance value
    assert Model(x=2) != Model()


def test_model_in_model():
    class Subsystem(System):
        x: Variable = Variable(default=1)

    class Model(System):
        x = Variable(default=1)
        s = Subsystem()

    assert not is_same_variable(Model.x, Model.s.x)

    class Model(System):
        x = Variable(default=1)
        s = Subsystem(x=x)

    assert is_same_variable(Model.x, Model.s.x)


def test_derivative():
    class Model(System):
        x: Variable = Variable(default=0)
        vx: Variable = x.derive(default=1)
        vx2: Variable = x.derive(default=1)
        vx2 = vx
        vx2: Variable = x.derive(default=2)

    model = Model(x=1, vx=2)
    assert is_derivative(model.vx, model.x)

    # Derivative of a "referenced" external variable
    class BigModel(System):
        x = Variable(default=1)
        m = Model(x=x)

    model = BigModel()
    assert is_same_variable(model.x, model.m.x)
    assert is_derivative(model.m.vx, model.m.x)
    assert is_derivative(model.m.vx, model.x)

    # Derivative of a "referenced" external variable
    class BigModel(System):
        x = Variable(default=1)
        vx = x.derive(default=2)
        m = Model(x=x)

    model = BigModel()
    assert is_same_variable(model.x, model.m.x)
    assert is_derivative(model.m.vx, model.m.x)
    assert is_derivative(model.m.vx, model.x)
    assert is_same_variable(model.vx, model.m.vx)

    # Assign derivative to external variable
    # which is not a derivative
    # breaks the relationship `vx = x.derivative` in `Model`
    with raises(TypeError):

        class BigModel3(System):
            vx = Variable(default=0)
            m = Model(vx=vx)

    # Assigning both variable and derivative to
    # external variable and its corresponding derivative
    # keeps the relationship inside `Model`
    class BigModel2(System):
        x = Variable(default=1)
        vx: float = x.derive(default=0)
        m = Model(x=x, vx=vx)
