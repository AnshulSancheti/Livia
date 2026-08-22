"""One-Euro filter for landmark streams. Same constants as Dart later."""

from __future__ import annotations

import math

from . import constants as C


class _LowPass:
    def __init__(self) -> None:
        self._hat = None

    def filter(self, x: float, alpha: float) -> float:
        if self._hat is None:
            self._hat = x
        else:
            self._hat = alpha * x + (1.0 - alpha) * self._hat
        return self._hat


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuro:
    def __init__(
        self,
        mincutoff: float = C.ONE_EURO_MINCUTOFF,
        beta: float = C.ONE_EURO_BETA,
        dcutoff: float = C.ONE_EURO_DCUTOFF,
    ) -> None:
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self._x = _LowPass()
        self._dx = _LowPass()
        self._last_t: float | None = None

    def filter(self, t: float, x: float) -> float:
        if x != x:  # NaN: hold last good estimate
            if self._x._hat is None:
                return float("nan")
            return self._x._hat
        if self._last_t is None:
            self._last_t = t
            return self._x.filter(x, 1.0)
        dt = max(t - self._last_t, 1e-6)
        self._last_t = t
        prev = self._x._hat if self._x._hat is not None else x
        dx = (x - prev) / dt
        edx = self._dx.filter(dx, _alpha(self.dcutoff, dt))
        cutoff = self.mincutoff + self.beta * abs(edx)
        return self._x.filter(x, _alpha(cutoff, dt))


class LandmarkSmoother:
    """Independent One-Euro on x, y, z per landmark."""

    def __init__(self, n: int = 33) -> None:
        self._filters = [[OneEuro(), OneEuro(), OneEuro()] for _ in range(n)]

    def apply(self, t: float, xyz: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
        out = []
        for i, (x, y, z) in enumerate(xyz):
            fx, fy, fz = self._filters[i]
            out.append((fx.filter(t, x), fy.filter(t, y), fz.filter(t, z)))
        return out
