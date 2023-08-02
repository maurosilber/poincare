from __future__ import annotations

from collections import ChainMap
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from symbolite import Symbol

from . import Constant, Derivative, Parameter, System, Variable, compile
from .types import Initial, Number


class RHS(Protocol):
    def __call__(self, t: float, y: NDArray, p: NDArray, dy: NDArray):
        ...


class Transform(Protocol):
    def __call__(self, t: float, y: NDArray, p: NDArray) -> NDArray:
        ...


def identity(t, y, p):
    return y


@dataclass
class Problem:
    rhs: RHS
    t: tuple[float, float]
    y: NDArray
    p: NDArray
    transform: Transform = identity


@dataclass
class Solution:
    t: NDArray
    y: NDArray


class Simulator:
    def __init__(
        self,
        system: System | type[System],
        /,
        *,
        backend=compile.Backend.FIRST_ORDER_VECTORIZED_NUMPY,
    ):
        self.model = system

        (
            _variable_names,
            _parameter_names,
            self._init_func,
            self._ode_func,
            self._param_func,
        ) = compile.compile(system, backend)

        def _name_to_object(system, name: str):
            for name in name[1:].split("."):
                if name.isdecimal():
                    system = Derivative(system, order=int(name))
                else:
                    system = getattr(system, name)
            return system

        self._parameter_map = {
            _name_to_object(system, n): n.removeprefix(".") for n in _parameter_names
        }
        self._variable_map = {
            _name_to_object(system, n): n.removeprefix(".") for n in _variable_names
        }

        self._defaults = {}
        for v in system._yield(Constant | Parameter):
            self._defaults[v] = v.default
        for v in system._yield(Variable | Derivative):
            self._defaults[v] = v.initial

    def create_problem(
        self,
        values: dict[Constant | Parameter | Variable | Derivative, Initial] = {},
        *,
        t_span: tuple[float, float] = (0, np.inf),
    ):
        values = self._resolve_initials(values)
        y0 = np.fromiter(
            (values[k] for k in self._variable_map),
            dtype=float,
            count=len(self._variable_map),
        )
        p0 = np.fromiter(
            (values[k] for k in self._parameter_map),
            dtype=float,
            count=len(self._parameter_map),
        )
        return Problem(self._ode_func, t_span, y0, p0)

    def solve(
        self,
        values: dict[Constant | Parameter | Variable | Derivative, Initial] = {},
        *,
        t_span: tuple[float, float] = (0, np.inf),
        times: NDArray,
    ):
        from scipy.integrate import odeint

        if t_span[0] != 0:
            raise NotImplementedError("odeint only works from t=0")

        problem = self.create_problem(values, t_span=t_span)

        def func(y, t, p, dy):
            self._ode_func(t, y, p, dy)
            return dy

        dy = np.empty_like(problem.y)
        result = odeint(func, problem.y, times, args=(problem.p, dy))
        return pd.DataFrame(result, columns=self._variable_map.values(), index=times)

    def _resolve_initials(
        self,
        values: dict[
            Constant | Parameter | Variable | Derivative, Initial | Symbol
        ] = {},
    ) -> dict[Constant | Parameter | Variable | Derivative, Number]:
        out: dict[Constant | Parameter | Variable | Derivative, Number] = {}
        defaults = ChainMap(
            out,  # not necessary, but caches previous solutions
            values,  # overrides defaults
            self._defaults,  # defaults
        )

        class Resolver:
            def get(self, key, default=None):
                if key in keys:
                    v = resolve(key)
                    return v
                else:
                    v = out.get(key, key)
                    return v

        resolver = Resolver()

        def resolve(k) -> Number:
            if k in processing:
                i = processing.index(k)
                processing.append(k)
                raise RecursionError(f"cyclic dependency chain: {processing[i:]}")
            processing.append(k)
            v = defaults[k]
            if not isinstance(v, Number):
                v = v.subs(resolver)
                if not isinstance(v, Number):
                    v = v.eval()
            out[k] = v
            keys.remove(k)
            assert processing[-1] == k
            del processing[-1]
            return v

        keys = set(defaults.keys())
        processing = []
        while keys:
            k = next(iter(keys))
            resolve(k)
        return out
