from __future__ import annotations

from collections import ChainMap
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from symbolite import Symbol

from . import Constant, Derivative, Parameter, System, Variable
from ._node import Node
from ._utils import eval_content
from .compile import (
    RHS,
    Array,
    Backend,
    Transform,
    compile_diffeq,
    compile_transform,
    depends_on_at_least_one_variable_or_time,
)
from .types import Initial, Number


@dataclass
class Problem:
    rhs: RHS
    t: tuple[float, float]
    y: Array
    p: Array
    transform: Transform


@dataclass
class Solution:
    t: Array
    y: Array


class Simulator:
    def __init__(
        self,
        system: System | type[System],
        /,
        *,
        backend: Backend = Backend.FIRST_ORDER_VECTORIZED_NUMPY,
        transform: dict[str, Symbol] | None = None,
    ):
        self.model = system
        self.compiled = compile_diffeq(system, backend)
        self.transform = compile_transform(system, self.compiled, transform)

    def create_problem(
        self,
        values: dict[
            Constant | Parameter | Variable | Derivative, Initial | Symbol
        ] = {},
        *,
        t_span: tuple[float, float] = (0, np.inf),
        transform=None,
    ):
        if transform is None:
            transform = self.transform
        elif transform is not self.transform:
            raise NotImplementedError("must recompile transform function")

        time = self.model.time
        if len(values.keys() - self.compiled.mapper) > 0 or any(
            depends_on_at_least_one_variable_or_time(v, time=time)
            for v in values.values()
        ):
            raise ValueError("must recompile to change assignments")

        content = ChainMap(values, self.compiled.mapper)
        assert self.compiled.libsl is not None
        result = eval_content(
            content,
            self.compiled.libsl,
            is_root=lambda x: isinstance(x, Number),
            is_dependency=lambda x: isinstance(x, Node),
        )
        y0 = np.fromiter(
            (result[k] for k in self.compiled.variables),
            dtype=float,
            count=len(self.compiled.variables),
        )
        p0 = np.fromiter(
            (result[k] for k in self.compiled.parameters),
            dtype=float,
            count=len(self.compiled.parameters),
        )
        return Problem(
            self.compiled.func,
            t_span,
            y0,
            p0,
            transform=transform.func,
        )

    def solve(
        self,
        values: dict[
            Constant | Parameter | Variable | Derivative, Initial | Symbol
        ] = {},
        *,
        t_span: tuple[float, float] = (0, np.inf),
        times: Array,
    ):
        from scipy.integrate import odeint

        if t_span[0] != 0:
            raise NotImplementedError("odeint only works from t=0")

        times = np.asarray(times)

        problem = self.create_problem(values, t_span=t_span)
        dy = np.empty_like(problem.y)
        result = odeint(
            problem.rhs,
            problem.y,
            times,
            args=(problem.p, dy),
            tfirst=True,
        )
        result = self.transform.func(
            times,
            result.T,
            problem.p,
            np.empty(
                (times.size, len(self.transform.output_names)),
                dtype=result.dtype,
            ).T,
        ).T
        return pd.DataFrame(
            result,
            columns=self.transform.output_names,
            index=pd.Series(times, name="time"),
        )

    def interact(
        self,
        values: dict[Constant | Parameter | Variable | Derivative, Initial] = {},
        *,
        t_span: tuple[float, float] = (0, np.inf),
        times: Array,
        func: Callable[[pd.DataFrame], Any] | None = None,
    ):
        import ipywidgets

        if len(values) == 0:
            values = {
                k: (0, 10, 0.1)
                for k, v in self.compiled.mapper.items()
                if isinstance(v, Initial)
            }

        name_map = {}
        value_map = {}
        for k, v in values.items():
            name = str(k)
            name_map[name] = k
            value_map[name] = v

        def solve_and_plot(**kwargs):
            result = self.solve(
                {name_map[k]: v for k, v in kwargs.items()},
                t_span=t_span,
                times=times,
            )
            if func is None:
                result.plot()
            else:
                func(result)

        ipywidgets.interact(solve_and_plot, **value_map)
