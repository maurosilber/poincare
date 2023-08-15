from __future__ import annotations

from collections import ChainMap
from dataclasses import dataclass

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
    Compiled,
    Transform,
    compile_diffeq,
    depends_on_at_least_one_variable_or_time,
)
from .types import Initial


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
    model = System | type[System]
    compiled: Compiled[Variable | Derivative, Parameter, RHS]
    transform: Compiled[Variable | Derivative, Parameter, Transform]

    def __init__(
        self,
        system: System | type[System],
        /,
        *,
        backend: Backend = Backend.FIRST_ORDER_VECTORIZED_NUMPY,
    ):
        self.model = system
        self.compiled = compile_diffeq(system, backend)

    def create_problem(
        self,
        values: dict[
            Constant | Parameter | Variable | Derivative, Initial | Symbol
        ] = {},
        *,
        t_span: tuple[float, float] = (0, np.inf),
    ):
        if len(values.keys() - self.compiled.mapper) > 0 or any(
            map(depends_on_at_least_one_variable_or_time, values.values())
        ):
            raise ValueError("must recompile to change assignments")

        content = ChainMap(values, self.compiled.mapper)
        assert self.compiled.libsl is not None
        result = eval_content(content, self.compiled.libsl, Node)
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
        return Problem(self.compiled.func, t_span, y0, p0)

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
        return pd.DataFrame(
            result,
            columns=self.compiled.output_names,
            index=pd.Series(times, name="time"),
        )
