import pytest
from symbolite import scalar
from symbolite.impl import libstd

from .._utils import eval_content


class SimpleVariable(scalar.Scalar):
    """Special type of Scalar that is evaluated to itself."""

    def __repr__(self):
        return self.name

    def eval(self, libsl=None):
        return self


class SimpleParameter(scalar.Scalar):
    """Special type of Scalar that is evaluated to itself."""

    def __repr__(self):
        return self.name

    def eval(self, libsl=None):
        return self


def is_root(x):
    return isinstance(x, scalar.NumberT)


def is_dependency(x):
    return isinstance(x, (SimpleParameter, SimpleVariable))


def test_eval_content():
    d = {SimpleParameter("x"): 1}

    assert eval_content(d, libstd, is_root, is_dependency) == {SimpleParameter("x"): 1}

    d = {SimpleParameter("x"): 1, SimpleParameter("y"): 2 * SimpleParameter("x")}

    assert eval_content(d, libstd, is_root, is_dependency) == {
        SimpleParameter("x"): 1,
        SimpleParameter("y"): 2,
    }


def test_cyclic():
    d = {
        SimpleParameter("x"): 2 * SimpleParameter("x"),
    }

    with pytest.raises(ValueError):
        assert eval_content(d, libstd, is_root, is_dependency)

    d = {
        SimpleParameter("x"): 2 * SimpleParameter("y"),
        SimpleParameter("y"): 2 * SimpleParameter("x"),
    }

    with pytest.raises(ValueError):
        assert eval_content(d, libstd, is_root, is_dependency)
