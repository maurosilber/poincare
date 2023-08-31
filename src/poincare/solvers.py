from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import integrate

if TYPE_CHECKING:
    from .simulator import Problem


__all__ = [
    "LSODA",
    "RK23",
    "RK45",
    "DOP853",
    "Radau",
    "BDF",
]


@dataclass
class Solution:
    t: NDArray
    y: NDArray


def _solve_scipy(
    problem: Problem,
    method: type[integrate.OdeSolver],
    options: dict,
    *,
    save_at: np.ndarray,
):
    dy = np.empty_like(problem.y)
    solution = integrate.solve_ivp(
        problem.rhs,
        (problem.t[0], min(problem.t[1], save_at[-1])),
        problem.y,
        method=method,
        t_eval=save_at,
        args=(problem.p, dy),
        **options,
    )
    out = np.empty(
        (solution.t.size, len(problem.scale)),
        dtype=solution.y.dtype,
    )
    out = problem.transform(
        solution.t,
        solution.y,
        problem.p,
        out.T,
    ).T
    return Solution(solution.t, out)


@dataclass(frozen=True, kw_only=True)
class _Base:
    # Relative and absolute tolerences
    rtol: float | Sequence[float] = 1e-3
    atol: float | Sequence[float] = 1e-6
    # Step size. By default, determined by the solver.
    first_step: float | None = None
    max_step: float = np.inf

    def __init_subclass__(cls, *, solver: type[integrate.OdeSolver]) -> None:
        cls._solver_class = solver
        assert cls.__name__ in __all__, cls.__name__

    def __call__(self, problem: Problem, *, save_at: np.ndarray):
        options = {k: getattr(self, k) for k in self.__dataclass_fields__}
        return _solve_scipy(
            problem,
            self._solver_class,
            options=options,
            save_at=save_at,
        )


@dataclass(frozen=True, kw_only=True)
class LSODA(_Base, solver=integrate.LSODA):
    """Adams/BDF method with automatic stiffness detection and switching.

    This is a wrapper to SciPy's LSODA, which in turn is a wrapper of ODEPACK's Fortran solver.
    """

    min_step: float = 0


@dataclass(frozen=True, kw_only=True)
class RK23(_Base, solver=integrate.RK23):
    """Explicit Runge-Kutta method of order 3(2).

    This is a wrapper to SciPy's RK23.
    """


@dataclass(frozen=True, kw_only=True)
class RK45(_Base, solver=integrate.RK45):
    """Explicit Runge-Kutta method of order 5(4).

    This is a wrapper to SciPy's RK45.
    """


@dataclass(frozen=True, kw_only=True)
class DOP853(_Base, solver=integrate.DOP853):
    """Explicit Runge-Kutta method of order 8.

    This is a wrapper to SciPy's DOP853.
    """


@dataclass(frozen=True, kw_only=True)
class Radau(_Base, solver=integrate.Radau):
    """Implicit Runge-Kutta method of Radau IIA family of order 5.

    This is a wrapper to SciPy's Radau.
    """


@dataclass(frozen=True, kw_only=True)
class BDF(_Base, solver=integrate.BDF):
    """Implicit method based on backward-differentiation formulas.

    This is a wrapper to SciPy's BDF.
    """
