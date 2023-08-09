from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

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
    compile,
    depends_on_at_least_one_variable_or_time,
)
from .types import Initial


class Transform(Protocol):
    def __call__(self, t: float, y: Array, p: Array) -> Array:
        ...


def identity(t: float, y: Array, p: Array) -> Array:
    return y


@dataclass
class Problem:
    rhs: RHS
    t: tuple[float, float]
    y: Array
    p: Array
    transform: Transform = identity


@dataclass
class Solution:
    t: Array
    y: Array


class Simulator:
    model = System | type[System]
    compiled = Compiled[RHS]

    _defaults: dict[Constant | Parameter | Variable | Derivative, Initial | Symbol]

    def __init__(
        self,
        system: System | type[System],
        /,
        *,
        backend: Backend = Backend.FIRST_ORDER_VECTORIZED_NUMPY,
    ):
        self.model = system
        self.compiled = compile(system, backend)
        self.variable_names = tuple(map(str, self.compiled.variables))

        self._defaults = {}
        self._default_functions = {}
        for v in system._yield(Constant | Parameter):
            if v.default is None:
                raise TypeError("Missing initial values. System must be instantiated.")
            elif isinstance(v, Parameter) and depends_on_at_least_one_variable_or_time(
                v.default
            ):
                self._default_functions[v] = v.default
            else:
                self._defaults[v] = v.default
        for v in system._yield(Variable | Derivative):
            if v.initial is None:
                raise TypeError("Missing initial values. System must be instantiated.")
            self._defaults[v] = v.initial

    def create_problem(
        self,
        values: dict[
            Constant | Parameter | Variable | Derivative, Initial | Symbol
        ] = {},
        *,
        t_span: tuple[float, float] = (0, np.inf),
    ):
        if not self._default_functions.keys().isdisjoint(values.keys()):
            raise NotImplementedError("must recompile to change assignments")

        content = {
            **self._defaults,
            **values,
        }

        assert self.compiled.libsl is not None
        result = eval_content(
            content,
            self.compiled.libsl,
            (
                # Scalar,  # time
                Node,  # rest
            ),
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
        return Problem(self.compiled.ode_func, t_span, y0, p0)

    def solve(
        self,
        values: dict[Constant | Parameter | Variable | Derivative, Initial] = {},
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
            columns=self.variable_names,
            index=pd.Series(times, name="time"),
        )
