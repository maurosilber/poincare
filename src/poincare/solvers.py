from __future__ import annotations

import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import integrate
from typing_extensions import assert_never

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

_cache = weakref.WeakKeyDictionary()


class Solver(Protocol):
    def __call__(self, problem: Problem, *, save_at) -> Solution:
        pass


@dataclass
class Solution:
    t: NDArray
    y: NDArray


def _solve_ivp_scipy(
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
        method=method,  # type: ignore
        t_eval=save_at,
        args=(problem.p, dy),
        **options,
    )
    if solution.status == -1:
        raise RuntimeError(solution.message)
    return _transform(problem, solution)


def _transform(problem: Problem, solution: Solution) -> Solution:
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


def _solve_numbalsoda(problem: Problem, solver, *, save_at: np.ndarray, atol, rtol):
    from numba import TypingError, cfunc
    from numbalsoda import lsoda_sig

    _rhs = problem.rhs
    try:
        rhs = _cache[_rhs]
    except KeyError:
        try:

            @cfunc(lsoda_sig)
            def rhs(t, u, du, p):
                _rhs(t, u, p, du)

            _cache[_rhs] = rhs
        except TypingError:
            raise TypingError("are you using Backend.NUMBA?") from None

    y, success = solver(
        rhs.address,
        problem.y,
        t_eval=save_at,
        data=problem.p,
        atol=atol,
        rtol=rtol,
    )
    if not success:
        raise RuntimeError("solver did not succeed.")
    return _transform(problem, Solution(save_at, y.T))


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
        return _solve_ivp_scipy(
            problem,
            self._solver_class,
            options={
                "rtol": self.rtol,
                "atol": self.atol,
                "first_step": self.first_step,
                "max_step": self.max_step,
            },
            save_at=save_at,
        )


@dataclass(frozen=True, kw_only=True)
class LSODA(_Base, solver=integrate.LSODA):
    """Adams/BDF method with automatic stiffness detection and switching.

    This is a wrapper to SciPy's LSODA, which in turn is a wrapper of ODEPACK's Fortran solver.
    """

    min_step: float = 0
    implementation: Literal["LSODA", "odeint", "numbalsoda", None] = None

    def __call__(self, problem: Problem, *, save_at: np.ndarray):
        match self.implementation:
            case "LSODA":
                return self._LSODA(problem, save_at=save_at)
            case "odeint":
                return self._odeint(problem, save_at=save_at)
            case "numbalsoda":
                from numbalsoda import lsoda

                return _solve_numbalsoda(
                    problem,
                    lsoda,
                    save_at=save_at,
                    rtol=self.rtol,
                    atol=self.atol,
                )
            case None:
                if problem.t[0] == 0:
                    return self._odeint(problem, save_at=save_at)
                else:
                    return self._LSODA(problem, save_at=save_at)
            case _:
                assert_never(self.implementation)

    def _LSODA(self, problem: Problem, *, save_at: np.ndarray):
        return _solve_ivp_scipy(
            problem,
            self._solver_class,
            options=dict(
                rtol=self.rtol,
                atol=self.atol,
                first_step=self.first_step,
                max_step=self.max_step,
                min_step=self.min_step,
            ),
            save_at=save_at,
        )

    def _odeint(self, problem: Problem, *, save_at: np.ndarray):
        dy = np.empty_like(problem.y)
        y = integrate.odeint(
            problem.rhs,
            tfirst=True,
            t=save_at,
            y0=problem.y,
            args=(problem.p, dy),
            atol=self.atol,
            rtol=self.rtol,
            h0=self.first_step if self.first_step is not None else 0,
            hmin=self.min_step,
            hmax=self.max_step if self.max_step is not np.inf else 0,
        )
        return _transform(problem, Solution(save_at, y.T))


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

    implementation: Literal["scipy", "numbalsoda"] = "scipy"

    def __call__(self, problem: Problem, *, save_at: np.ndarray):
        match self.implementation:
            case "scipy":
                return super().__call__(problem, save_at=save_at)
            case "numbalsoda":
                from numbalsoda import dop853

                return _solve_numbalsoda(
                    problem,
                    dop853,
                    save_at=save_at,
                    rtol=self.rtol,
                    atol=self.atol,
                )
            case _:
                assert_never(self.implementation)


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
