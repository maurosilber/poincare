from __future__ import annotations

from collections import ChainMap
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from symbolite import Symbol

from . import Constant, Derivative, Parameter, System, Variable
from .compile import RHS, Array, Backend, Compiled, compile
from .types import Initial, Number


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

    def __init__(
        self,
        system: System | type[System],
        /,
        *,
        backend: Backend = Backend.FIRST_ORDER_VECTORIZED_NUMPY,
    ):
        self.model = system
        self.compiled = compile(system, backend)

        def _name_to_object(system, name: str):
            for name in name[1:].split("."):
                if name.isdecimal():
                    system = Derivative(system, order=int(name))
                else:
                    system = getattr(system, name)
            return system

        self._parameter_map = {
            _name_to_object(system, n): n.removeprefix(".")
            for n in self.compiled.parameter_names
        }
        self._variable_map = {
            _name_to_object(system, n): n.removeprefix(".")
            for n in self.compiled.variable_names
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
            columns=self._variable_map.values(),
            index=pd.Series(times, name="time"),
        )

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
