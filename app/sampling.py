"""RAS -> voxel mapping and trilinear interpolation."""
from __future__ import annotations

import math

import numpy as np

from .errors import PointError
from .nifti import NiftiImage

# Tolerance (in voxel units) absorbing floating-point noise from the inverse
# affine before the closed voxel-center domain [0, n-1] is enforced.
DOMAIN_EPS = 1e-6


def sample_point(img: NiftiImage, ras: list[float]) -> tuple[list[float], float]:
    """Sample ``img`` at RAS ``(x, y, z)``.

    Returns the continuous voxel coordinate ``(i, j, k)`` and the trilinearly
    interpolated, scaling-corrected intensity.  Raises :class:`PointError`
    for out-of-bounds coordinates or non-finite neighbour data.
    """
    point = img.inv_affine @ np.array([ras[0], ras[1], ras[2], 1.0])
    coords = (float(point[0]), float(point[1]), float(point[2]))
    dims = (img.nx, img.ny, img.nz)

    for axis, (c, n) in enumerate(zip(coords, dims)):
        name = "ijk"[axis]
        if not math.isfinite(c):
            raise PointError(
                "OUT_OF_BOUNDS",
                f"RAS point maps to a non-finite voxel coordinate on axis {name}",
            )
        if c < -DOMAIN_EPS or c > (n - 1) + DOMAIN_EPS:
            raise PointError(
                "OUT_OF_BOUNDS",
                f"voxel coordinate {name}={c:.6f} lies outside the closed "
                f"voxel-center domain [0, {n - 1}] (axis length n={n})",
            )
    clamped = [min(max(c, 0.0), float(n - 1)) for c, n in zip(coords, dims)]

    (i0, i1, fi), (j0, j1, fj), (k0, k1, fk) = (
        _axis_weights(clamped[0], img.nx),
        _axis_weights(clamped[1], img.ny),
        _axis_weights(clamped[2], img.nz),
    )

    total = 0.0
    for k, wk in ((k0, 1.0 - fk), (k1, fk)):
        for j, wj in ((j0, 1.0 - fj), (j1, fj)):
            for i, wi in ((i0, 1.0 - fi), (i1, fi)):
                # Scale first, then interpolate, per the platform contract.
                val = float(img.data[k, j, i]) * img.scl_slope + img.scl_inter
                if not math.isfinite(val):
                    raise PointError(
                        "NON_FINITE_DATA",
                        f"non-finite scaled intensity at neighbour voxel ({i}, {j}, {k})",
                    )
                total += wk * wj * wi * val

    voxel = [round(c, 9) for c in clamped]
    return voxel, total


def _axis_weights(c: float, n: int) -> tuple[int, int, float]:
    """Neighbour indices and fractional weight for one axis.

    ``c`` is already clamped to ``[0, n - 1]``.  On the upper boundary (or
    when the axis has a single slice) both neighbours collapse onto the
    unique endpoint and the fractional weight is zeroed.
    """
    lo = int(math.floor(c))
    hi = lo + 1
    if hi > n - 1:
        hi = n - 1
        lo = min(lo, n - 1)
    frac = c - lo
    if hi == lo:
        frac = 0.0
    return lo, hi, frac
