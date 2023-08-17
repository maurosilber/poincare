from __future__ import annotations

from collections import ChainMap
from dataclasses import dataclass
from typing import Any, Callable, Hashable, Mapping, Sequence

import numpy as np
import pandas as pd
import pint
import pint_pandas
from numpy.typing import ArrayLike
from symbolite import Symbol

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
from .types import Constant, Derivative, Initial, Number, Parameter, System, Variable


@dataclass
class Problem:
    rhs: RHS
    t: tuple[float, float]
    y: dict[Symbol, Number | pint.Quantity]
    p: dict[Symbol, Number | pint.Quantity]
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
        transform: Sequence[Symbol] | Mapping[Hashable, Symbol] | None = None,
    ):
        self.model = system
        self.compiled = compile_diffeq(system, backend)
        if isinstance(transform, Sequence):
            transform = {str(x): x for x in transform}
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

        for k, v in values.items():
            default = self.compiled.mapper[k]
            match [v, default]:
                case [pint.Quantity() as q1, pint.Quantity() as q2]:
                    if not q1.is_compatible_with(q2):
                        raise pint.DimensionalityError(
                            q1.units, q2.units, extra_msg=f" for {k}"
                        )
                case [pint.Quantity() as q, _] | [_, pint.Quantity() as q]:
                    if not q.dimensionless:
                        raise pint.DimensionalityError(
                            q.units, pint.Unit("dimensionless"), extra_msg=f" for {k}"
                        )

        content = ChainMap(values, self.compiled.mapper)
        assert self.compiled.libsl is not None
        result = eval_content(
            content,
            self.compiled.libsl,
            is_root=lambda x: isinstance(x, Number | pint.Quantity),
            is_dependency=lambda x: isinstance(x, Node),
        )
        y0 = {k: result[k] for k in self.compiled.variables}
        p0 = {k: result[k] for k in self.compiled.parameters}
        return Problem(self.compiled.func, t_span, y0, p0, transform=transform.func)

    def solve(
        self,
        values: dict[
            Constant | Parameter | Variable | Derivative, Initial | Symbol
        ] = {},
        *,
        t_span: tuple[float, float] = (0, np.inf),
        times: ArrayLike,
    ):
        from scipy.integrate import odeint

        if t_span[0] != 0:
            raise NotImplementedError("odeint only works from t=0")

        times = np.asarray(times)

        def to_magnitude(x):
            if isinstance(x, pint.Quantity):
                return x.to_base_units().magnitude
            else:
                return x

        def from_magnitude(x, x0):
            if isinstance(x0, pint.Quantity):
                unit = x0.units
                scale = (1 * x0.to_base_units().units).m_as(unit)
                if isinstance(x, np.ndarray):
                    return pint_pandas.PintArray(
                        x * scale,
                        pint_pandas.PintType(unit),
                    )
                else:
                    return (x * scale).to(unit)
            else:
                return x

        problem = self.create_problem(values, t_span=t_span)
        y = np.fromiter(
            map(to_magnitude, problem.y.values()),
            dtype=float,
            count=len(problem.y),
        )
        p = np.fromiter(
            map(to_magnitude, problem.p.values()),
            dtype=float,
            count=len(problem.p),
        )
        dy = np.empty_like(y)
        result = odeint(
            problem.rhs,
            y,
            times,
            args=(p, dy),
            tfirst=True,
        )
        result = self.transform.func(
            times,
            result.T,
            p,
            np.empty(
                (times.size, len(self.transform.output)),
                dtype=result.dtype,
            ).T,
        ).T

        output_units = eval_content(
            ChainMap(self.transform.output, problem.p, problem.y),
            self.transform.libsl,
            is_root=lambda x: isinstance(x, Number | pint.Quantity),
            is_dependency=lambda x: isinstance(x, Node),
        )
        output_units = {k: output_units[k] for k in self.transform.output.keys()}

        return pd.DataFrame(
            {
                k: from_magnitude(v, v0)
                for (k, v0), v in zip(output_units.items(), result.T)
            },
            index=pd.Series(times, name="time"),
        )

    def interact(
        self,
        values: dict[
            Constant | Parameter | Variable | Derivative, tuple[float, ...]
        ] = {},
        *,
        t_span: tuple[float, float] = (0, np.inf),
        times: ArrayLike,
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
