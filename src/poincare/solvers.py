from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy import integrate

if TYPE_CHECKING:
    from .simulator import Problem


@dataclass
class Solution:
    t: NDArray
    y: NDArray


def odeint(problem: Problem, *, save_at: np.ndarray):
    if problem.t[0] != 0:
        raise NotImplementedError("odeint only works from t=0")

    dy = np.empty_like(problem.y)
    y: NDArray[np.floating] = integrate.odeint(
        problem.rhs,
        problem.y,
        save_at,
        args=(problem.p, dy),
        tfirst=True,
    )
    out = np.empty(
        (save_at.size, len(problem.scale)),
        dtype=y.dtype,
    )
    out = problem.transform(
        save_at,
        y.T,
        problem.p,
        out.T,
    ).T
    return Solution(save_at, out)
