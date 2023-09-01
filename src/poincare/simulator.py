from __future__ import annotations

import numbers
from collections import ChainMap
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Hashable, Mapping, Sequence

import numpy as np
import pandas as pd
import pint
import pint_pandas
from numpy.typing import ArrayLike
from symbolite import Symbol

from . import solvers
from ._node import Node
from ._utils import eval_content
from .compile import (
    RHS,
    Backend,
    Transform,
    compile_diffeq,
    compile_transform,
    depends_on_at_least_one_variable_or_time,
)
from .types import (
    Constant,
    Derivative,
    Initial,
    Number,
    Parameter,
    System,
    Variable,
)

if TYPE_CHECKING:
    import ipywidgets

Components = Constant | Parameter | Variable | Derivative


@dataclass
class Problem:
    rhs: RHS
    t: tuple[float, float]
    y: Sequence[Number]
    p: Sequence[Number]
    transform: Transform
    scale: Sequence[Number | pint.Quantity]


def rescale(q: Number | pint.Quantity) -> Number:
    if isinstance(q, pint.Quantity):
        return q.to_base_units().magnitude
    else:
        return q


def get_scale(q: Number | pint.Quantity) -> Number | pint.Quantity:
    if isinstance(q, pint.Quantity):
        unit = q.units
        scale = (1 / unit).to_base_units().magnitude
        return scale * unit
    else:
        return 1


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
        self.transform = self._compile_transform(transform)

    def _compile_transform(
        self,
        transform: Sequence[Symbol] | Mapping[Hashable, Symbol] | None,
    ):
        if isinstance(transform, Sequence):
            transform = {str(x): x for x in transform}
        return compile_transform(self.model, self.compiled, transform)

    def create_problem(
        self,
        values: Mapping[Components, Initial | Symbol] = {},
        *,
        t_span: tuple[float, float] = (0, np.inf),
        transform: Sequence[Symbol] | Mapping[Hashable, Symbol] | None = None,
    ):
        if transform is None:
            compiled_transform = self.transform
        else:
            compiled_transform = self._compile_transform(transform)

        if any(
            depends_on_at_least_one_variable_or_time(self.compiled.mapper[k])
            or depends_on_at_least_one_variable_or_time(v)
            for k, v in values.items()
        ):
            raise ValueError("must recompile to change time-dependent assignments")

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

        content = ChainMap(
            values,
            self.compiled.mapper,
            self.transform.output,
            {self.compiled.independent[0]: t_span[0]},
        )
        assert self.compiled.libsl is not None
        result = eval_content(
            content,
            self.compiled.libsl,
            is_root=lambda x: isinstance(x, Number | pint.Quantity),
            is_dependency=lambda x: isinstance(x, Node),
        )
        y0 = np.fromiter(
            map(rescale, (result[k] for k in self.compiled.variables)),
            dtype=float,
            count=len(self.compiled.variables),
        )
        p0 = np.fromiter(
            map(rescale, (result[k] for k in self.compiled.parameters)),
            dtype=float,
            count=len(self.compiled.parameters),
        )
        scale = [get_scale(result[k]) for k in self.transform.output.keys()]
        return Problem(
            rhs=self.compiled.func,
            t=t_span,
            y=y0,
            p=p0,
            transform=compiled_transform.func,
            scale=scale,
        )

    def solve(
        self,
        values: Mapping[Components, Initial | Symbol] = {},
        *,
        t_span: tuple[float, float] = (0, np.inf),
        save_at: Sequence[float],
        solver: solvers.Solver = solvers.LSODA(),
    ):
        problem = self.create_problem(values, t_span=t_span)
        solution = solver(problem, save_at=np.asarray(save_at))
        return pd.DataFrame(
            {
                k: pint_pandas.PintArray(x * s.magnitude, pint_pandas.PintType(s.units))
                if isinstance(s, pint.Quantity)
                else x * s
                for k, s, x in zip(
                    self.transform.output.keys(), problem.scale, solution.y.T
                )
            },
            index=pd.Series(solution.t, name="time"),
        )

    def interact(
        self,
        values: Mapping[Components, tuple[float, ...] | ipywidgets.Widget]
        | Sequence[Components] = {},
        *,
        t_span: tuple[float, float] = (0, np.inf),
        times: ArrayLike,
        func: Callable[[pd.DataFrame], Any] = lambda df: df.plot(),
    ):
        try:
            import ipywidgets
        except ImportError:
            raise ImportError(
                "must install ipywidgets to use interactuve."
                " Run `pip install ipywidgets`."
            )

        if len(values) == 0:
            values = self.compiled.mapper
        elif isinstance(values, Sequence):
            values = {k: self.compiled.mapper[k] for k in values}

        name_map = {}
        value_map = {}
        unit_map = {}
        for k, v in values.items():
            unit = 1
            match v:
                case numbers.Real() as default:
                    if v == 0:
                        v = 1
                    widget = ipywidgets.FloatSlider(
                        default,
                        min=default / 10,
                        max=v * 10,
                        step=0.1 * v,
                    )
                case pint.Quantity(magnitude=v, units=unit):
                    widget = ipywidgets.FloatSlider(
                        v, min=v / 10, max=v * 10, step=0.1 * v
                    )
                case (min, max, step):
                    v = self.compiled.mapper[k]
                    if not isinstance(v, Number):
                        v = None
                    widget = ipywidgets.FloatSlider(v, min=min, max=max, step=step)
                case ipywidgets.Widget():
                    widget = v
                case _:
                    continue

            name = str(k)
            name_map[name] = k
            value_map[name] = widget
            unit_map[name] = unit

        def solve_and_plot(**kwargs):
            result = self.solve(
                {name_map[k]: v * unit_map.get(k, 1) for k, v in kwargs.items()},
                t_span=t_span,
                save_at=times,
            )
            func(result)

        ipywidgets.interact(solve_and_plot, **value_map)
