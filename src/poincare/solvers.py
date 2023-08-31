from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import integrate

if TYPE_CHECKING:
    from .simulator import Problem


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
        problem.t,
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


@dataclass(frozen=True, slots=True, kw_only=True)
class LSODA:
    """Adams/BDF method with automatic stiffness detection and switching.

    This is a wrapper to SciPy's LSODA, which in turn is a wrapper of ODEPACK's Fortran solver.
    """

    # Relative and absolute tolerences
    rtol: float | Sequence[float] = 1e-3
    atol: float | Sequence[float] = 1e-6
    # Step size. By default, determined by the solver.
    first_step: float | None = None
    min_step: float = 0
    max_step: float = np.inf

    def __call__(self, problem: Problem, *, save_at: np.ndarray):
        return _solve_scipy(
            problem,
            integrate.LSODA,
            options={
                "rtol": self.rtol,
                "atol": self.atol,
                "first_step": self.first_step,
                "min_step": self.min_step,
                "max_step": self.max_step,
            },
            save_at=save_at,
        )
