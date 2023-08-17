from pathlib import Path
from runpy import run_path

import matplotlib.pyplot as plt
from pytest import mark

path = Path("examples")
examples = [str(p) for p in path.glob("*.py") if not p.name.startswith("test_")]


@mark.parametrize("example", examples)
def test_example(example, monkeypatch):
    monkeypatch.setattr(plt, "show", noop)
    run_path(example, run_name="__main__")


def noop(*args, **kwargs):
    pass
