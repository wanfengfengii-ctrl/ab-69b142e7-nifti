"""Synthesise NIfTI-1 .nii byte strings for tests and smoke checks.

The builder always emits a header whose declared layout matches the bytes it
produces (unless the caller deliberately breaks it via the keyword
arguments), so each validation rule can be exercised in isolation.
"""
from __future__ import annotations

import math
import struct

import numpy as np

DT_INT16 = 4
DT_FLOAT32 = 16
DT_UINT8 = 2
DT_INT32 = 8
DT_FLOAT64 = 64

_NP_KIND = {DT_INT16: "i2", DT_FLOAT32: "f4", DT_UINT8: "u1", DT_INT32: "i4", DT_FLOAT64: "f8"}
_BITPIX = {DT_INT16: 16, DT_FLOAT32: 32, DT_UINT8: 8, DT_INT32: 32, DT_FLOAT64: 64}


def default_volume(dims: tuple[int, int, int]) -> np.ndarray:
    """(nz, ny, nx) float64 volume with value i + 10j + 100k at (i, j, k)."""
    nx, ny, nz = dims
    i, j, k = np.meshgrid(
        np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"
    )
    return (i + 10 * j + 100 * k).astype(np.float64).transpose(2, 1, 0).copy()


def build_nifti(
    *,
    endian: str = "<",
    datatype: int = DT_INT16,
    dims: tuple[int, int, int] = (4, 4, 4),
    data: np.ndarray | None = None,
    sform: list[list[float]] | None = None,
    sform_code: int | None = None,
    qform: dict | None = None,
    qform_code: int | None = None,
    slope: float = 1.0,
    inter: float = 0.0,
    vox_offset: float = 352.0,
    bitpix: int | None = None,
    magic: bytes = b"n+1\x00",
    sizeof_hdr: int = 348,
    trailing: bytes = b"",
    drop: int = 0,
) -> bytes:
    """Build a .nii byte string.

    ``sform`` is a 3x4 matrix; ``qform`` is a dict with keys ``quat``
    ((b, c, d)), ``offset`` ((x, y, z)), ``pixdim`` ((p1, p2, p3)) and
    optional ``qfac`` (+1/-1, stored in pixdim[0]).  ``data`` is a
    (nz, ny, nx) array; when omitted, :func:`default_volume` is used.
    ``drop`` removes that many bytes from the end (truncation tests) and
    ``trailing`` appends extra bytes (trailing-byte tests).
    """
    nx, ny, nz = dims
    np_dtype = np.dtype(endian + _NP_KIND[datatype])
    if bitpix is None:
        bitpix = _BITPIX[datatype]

    if sform is not None and sform_code is None:
        sform_code = 1
    if qform is not None and qform_code is None:
        qform_code = 1
    sform_code = sform_code or 0
    qform_code = qform_code or 0

    hdr = bytearray(348)
    struct.pack_into(endian + "i", hdr, 0, sizeof_hdr)
    struct.pack_into(endian + "8h", hdr, 40, 3, nx, ny, nz, 1, 1, 1, 1)
    struct.pack_into(endian + "h", hdr, 70, datatype)
    struct.pack_into(endian + "h", hdr, 72, bitpix)

    pixdim = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    if qform is not None:
        pixdim[0] = float(qform.get("qfac", 1.0))
        pixdim[1], pixdim[2], pixdim[3] = (float(v) for v in qform["pixdim"])
    struct.pack_into(endian + "8f", hdr, 76, *pixdim)

    struct.pack_into(endian + "f", hdr, 108, float(vox_offset))
    struct.pack_into(endian + "f", hdr, 112, float(slope))
    struct.pack_into(endian + "f", hdr, 116, float(inter))
    struct.pack_into(endian + "h", hdr, 252, qform_code)
    struct.pack_into(endian + "h", hdr, 254, sform_code)

    if qform is not None:
        struct.pack_into(endian + "3f", hdr, 256, *(float(v) for v in qform["quat"]))
        struct.pack_into(endian + "3f", hdr, 268, *(float(v) for v in qform["offset"]))
    if sform is not None:
        for row, off in zip(sform, (280, 296, 312)):
            struct.pack_into(endian + "4f", hdr, off, *(float(v) for v in row))
    hdr[344:348] = magic

    if data is None:
        data = default_volume(dims)
    payload = np.asarray(data).astype(np_dtype).tobytes(order="C")

    pad = b"\x00" * max(0, int(vox_offset) - 352) if math.isfinite(vox_offset) else b""
    blob = bytes(hdr) + b"\x00" * 4 + pad + payload + trailing
    if drop:
        blob = blob[:-drop]
    return blob
