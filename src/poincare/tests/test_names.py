from .. import Constant, Derivative, Parameter, System, Variable, assign, initial


def test_single_level():
    class Model(System):
        c: Constant = assign(default=0, constant=True)
        p: Parameter = assign(default=0)
        v: Variable = initial(default=0)
        dv: Derivative = v.derive(initial=0)

    for m in [Model, Model()]:
        m: Model

        assert m.c.name == "c"
        assert m.p.name == "p"
        assert m.v.name == "v"
        assert m.dv.name == "dv"


def test_multi_level():
    class Level1(System):
        c1: Constant = assign(default=0, constant=True)
        p1: Parameter = assign(default=0)
        v1: Variable = initial(default=0)
        dv1: Derivative = v1.derive(initial=0)

    class Level0(System):
        c: Constant = assign(default=0, constant=True)
        p: Parameter = assign(default=0)
        v: Variable = initial(default=0)
        dv: Derivative = v.derive(initial=0)

        level1 = Level1(c1=c, p1=p, v1=v)

    for m in [Level0, Level0()]:
        m: Level0

        assert m.c.name == "c"
        assert m.p.name == "p"
        assert m.v.name == "v"
        assert m.dv.name == "dv"

        assert m.level1.c1.name == "c"
        assert m.level1.p1.name == "p"
        assert m.level1.v1.name == "v"
        assert m.level1.dv1.name == "dv"
