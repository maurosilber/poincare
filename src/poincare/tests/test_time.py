from ..types import Independent, Parameter, System, Variable, assign, initial


def test_unique_time():
    class SubModel(System):
        time = Independent(default=0)
        x: Variable = initial(default=0)
        eq = x.derive() << time

    class Model(System):
        time = Independent(default=0)
        sub = SubModel()

    assert Model.time is Model.sub.time


def test_parameter_time():
    class SubModel(System):
        time = Independent(default=0)
        p: Parameter = assign(default=time)

    class Model(System):
        time = Independent(default=0)
        sub = SubModel(p=0)

    assert Model.time is Model.sub.time
