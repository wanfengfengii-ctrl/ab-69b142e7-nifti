"""Strict NIfTI-1 single-file (.nii) parsing.

Only complete 3D ``.nii`` files carrying int16 or float32 payloads (either
byte order) are accepted.  The parser validates, and refuses to guess about:

* ``sizeof_hdr`` — fixes the byte order (little or big endian);
* ``magic`` — must be ``n+1\\0`` (single-file .nii, not a .hdr/.img pair);
* ``dim`` — exactly 3 dimensions, every axis length >= 1;
* ``datatype``/``bitpix`` — int16 (4/16) or float32 (16/32), consistent pair;
* ``vox_offset`` — finite, integer-valued, >= 352, and the file length must
  equal ``vox_offset + nvox * (bitpix/8)`` exactly (no truncation, no
  trailing bytes);
* ``scl_slope``/``scl_inter`` — finite (a zero slope means "no scaling" per
  the NIfTI-1 spec and is normalised to 1/0);
* transforms — a valid sform (``sform_code > 0``) takes priority over a valid
  qform; if neither is present, or the selected affine is not invertible,
  the file is rejected.

Every rejection raises :class:`~app.errors.ApiError` whose ``field`` names
the header field responsible.
"""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass

import numpy as np

from .errors import ApiError

SIZEOF_HDR = 348
EXTENDER_SIZE = 4
MIN_VOX_OFFSET = SIZEOF_HDR + EXTENDER_SIZE  # 352 for a .nii single file
MAGIC_NII = b"n+1\x00"

DT_INT16 = 4
DT_FLOAT32 = 16
_DTYPES = {DT_INT16: ("i2", 16), DT_FLOAT32: ("f4", 32)}

MAX_FILE_BYTES = 16 * 1024 * 1024  # 16 MiB

# NIfTI-1 header field offsets (bytes).
_OFF_DIM = 40
_OFF_DATATYPE = 70
_OFF_BITPIX = 72
_OFF_PIXDIM = 76
_OFF_VOX_OFFSET = 108
_OFF_SCL_SLOPE = 112
_OFF_SCL_INTER = 116
_OFF_QFORM_CODE = 252
_OFF_SFORM_CODE = 254
_OFF_QUATERN = 256  # b, c, d (3 x f32)
_OFF_QOFFSET = 268  # x, y, z (3 x f32)
_OFF_SROW_X = 280
_OFF_SROW_Y = 296
_OFF_SROW_Z = 312
_OFF_MAGIC = 344


@dataclass
class NiftiImage:
    """A validated 3D NIfTI volume plus its voxel->RAS affine."""

    nx: int
    ny: int
    nz: int
    datatype: int
    endian: str  # '<' little-endian, '>' big-endian
    affine: np.ndarray  # (4, 4) voxel (i,j,k) -> RAS (x,y,z)
    inv_affine: np.ndarray  # (4, 4) RAS -> voxel
    transform_source: str  # 'sform' | 'qform'
    scl_slope: float
    scl_inter: float
    data: np.ndarray  # shape (nz, ny, nx), header byte order preserved

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.nx, self.ny, self.nz)


def _i16(buf: bytes, off: int, en: str) -> int:
    return struct.unpack_from(en + "h", buf, off)[0]


def _f32(buf: bytes, off: int, en: str) -> float:
    return struct.unpack_from(en + "f", buf, off)[0]


def parse_nifti(buf: bytes) -> NiftiImage:
    """Parse and strictly validate a single-file NIfTI-1 .nii buffer."""
    if len(buf) > MAX_FILE_BYTES:
        raise ApiError(
            "FILE_TOO_LARGE",
            f"file is {len(buf)} bytes; the limit is {MAX_FILE_BYTES} bytes (16 MiB)",
            field="file",
            status=413,
        )
    if len(buf) < SIZEOF_HDR:
        raise ApiError(
            "HEADER_TRUNCATED",
            f"file is {len(buf)} bytes; a NIfTI-1 header needs {SIZEOF_HDR} bytes",
            field="file",
        )

    # Byte order is pinned down by sizeof_hdr.
    if struct.unpack_from("<i", buf, 0)[0] == SIZEOF_HDR:
        en = "<"
    elif struct.unpack_from(">i", buf, 0)[0] == SIZEOF_HDR:
        en = ">"
    else:
        raise ApiError(
            "BAD_SIZEOF_HDR",
            "sizeof_hdr must be 348 read as little- or big-endian int32; "
            f"got {struct.unpack_from('<i', buf, 0)[0]} (little-endian read)",
            field="sizeof_hdr",
        )

    magic = bytes(buf[_OFF_MAGIC : _OFF_MAGIC + 4])
    if magic != MAGIC_NII:
        raise ApiError(
            "BAD_MAGIC",
            f"magic must be b'n+1\\0' for a complete single-file .nii; got {magic!r}",
            field="magic",
        )

    dims = struct.unpack_from(en + "8h", buf, _OFF_DIM)
    if dims[0] != 3:
        raise ApiError(
            "BAD_DIM",
            f"dim[0] must be 3 for a 3D volume; got {dims[0]}",
            field="dim",
        )
    nx, ny, nz = dims[1], dims[2], dims[3]
    for axis, n in ((1, nx), (2, ny), (3, nz)):
        if n < 1:
            raise ApiError(
                "BAD_DIM",
                f"dim[{axis}] must be >= 1; got {n}",
                field="dim",
            )

    datatype = _i16(buf, _OFF_DATATYPE, en)
    if datatype not in _DTYPES:
        raise ApiError(
            "UNSUPPORTED_DATATYPE",
            f"datatype must be 4 (int16) or 16 (float32); got {datatype}",
            field="datatype",
        )
    kind, bitpix = _DTYPES[datatype]
    actual_bitpix = _i16(buf, _OFF_BITPIX, en)
    if actual_bitpix != bitpix:
        raise ApiError(
            "BITPIX_MISMATCH",
            f"bitpix must be {bitpix} for datatype {datatype}; got {actual_bitpix}",
            field="bitpix",
        )

    vox_offset_f = _f32(buf, _OFF_VOX_OFFSET, en)
    if (
        not math.isfinite(vox_offset_f)
        or vox_offset_f < MIN_VOX_OFFSET
        or vox_offset_f != math.floor(vox_offset_f)
    ):
        raise ApiError(
            "BAD_VOX_OFFSET",
            f"vox_offset must be a finite integer >= {MIN_VOX_OFFSET}; got {vox_offset_f}",
            field="vox_offset",
        )
    vox_offset = int(vox_offset_f)

    nvox = nx * ny * nz
    bytes_per_vox = bitpix // 8
    expected = vox_offset + nvox * bytes_per_vox
    if len(buf) < expected:
        raise ApiError(
            "PAYLOAD_TRUNCATED",
            f"payload truncated: header implies {expected} bytes "
            f"(vox_offset {vox_offset} + {nvox} voxels x {bytes_per_vox} B) "
            f"but the file has {len(buf)} bytes",
            field="file",
            details={"expected_bytes": expected, "actual_bytes": len(buf)},
        )
    if len(buf) > expected:
        raise ApiError(
            "TRAILING_BYTES",
            f"file has {len(buf) - expected} trailing byte(s): header implies "
            f"{expected} bytes but the file has {len(buf)} bytes",
            field="file",
            details={"expected_bytes": expected, "actual_bytes": len(buf)},
        )

    scl_slope = _f32(buf, _OFF_SCL_SLOPE, en)
    scl_inter = _f32(buf, _OFF_SCL_INTER, en)
    if not math.isfinite(scl_slope):
        raise ApiError(
            "NON_FINITE_SCALING",
            f"scl_slope must be finite; got {scl_slope}",
            field="scl_slope",
        )
    if not math.isfinite(scl_inter):
        raise ApiError(
            "NON_FINITE_SCALING",
            f"scl_inter must be finite; got {scl_inter}",
            field="scl_inter",
        )
    if scl_slope == 0.0:
        # NIfTI-1: a zero slope means the data is not scaled.
        scl_slope, scl_inter = 1.0, 0.0

    affine, source, source_field = _select_affine(buf, en)
    inv_affine = _invert_affine(affine, source, source_field)

    dtype = np.dtype(en + kind)
    data = np.frombuffer(buf, dtype=dtype, count=nvox, offset=vox_offset).reshape(
        (nz, ny, nx)
    )

    return NiftiImage(
        nx=nx,
        ny=ny,
        nz=nz,
        datatype=datatype,
        endian=en,
        affine=affine,
        inv_affine=inv_affine,
        transform_source=source,
        scl_slope=scl_slope,
        scl_inter=scl_inter,
        data=data,
    )


def _select_affine(buf: bytes, en: str) -> tuple[np.ndarray, str, str]:
    """Pick the voxel->RAS affine: a valid sform wins over the qform."""
    qform_code = _i16(buf, _OFF_QFORM_CODE, en)
    sform_code = _i16(buf, _OFF_SFORM_CODE, en)

    if sform_code > 0:
        rows = []
        for name, off in (
            ("srow_x", _OFF_SROW_X),
            ("srow_y", _OFF_SROW_Y),
            ("srow_z", _OFF_SROW_Z),
        ):
            row = struct.unpack_from(en + "4f", buf, off)
            if not all(math.isfinite(v) for v in row):
                raise ApiError(
                    "NON_FINITE_AFFINE",
                    f"{name} entries must be finite; got {list(row)}",
                    field=name,
                )
            rows.append(row)
        affine = np.array(
            [rows[0], rows[1], rows[2], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64
        )
        return affine, "sform", "srow_x"

    if qform_code > 0:
        return _qform_affine(buf, en), "qform", "pixdim"

    raise ApiError(
        "NO_VALID_TRANSFORM",
        "neither sform nor qform is present "
        f"(sform_code={sform_code}, qform_code={qform_code})",
        field="sform_code",
    )


def _qform_affine(buf: bytes, en: str) -> np.ndarray:
    """Rebuild the qform affine per nifti1_io.c nifti_quatern_to_mat44."""
    pixdim = struct.unpack_from(en + "8f", buf, _OFF_PIXDIM)
    b, c, d = struct.unpack_from(en + "3f", buf, _OFF_QUATERN)
    qx, qy, qz = struct.unpack_from(en + "3f", buf, _OFF_QOFFSET)

    for field, values in (
        ("pixdim", pixdim[:4]),
        ("quatern_b", (b, c, d)),
        ("qoffset_x", (qx, qy, qz)),
    ):
        if not all(math.isfinite(v) for v in values):
            raise ApiError(
                "NON_FINITE_AFFINE",
                f"{field} entries must be finite; got {list(values)}",
                field=field,
            )

    qfac = -1.0 if pixdim[0] < 0 else 1.0

    ss = b * b + c * c + d * d
    if 1.0 - ss < 1e-7:
        # Quaternion (b,c,d) is not a unit quaternion: normalise it.
        norm = math.sqrt(ss)
        if norm == 0.0:  # unreachable (ss >= 1 - 1e-7 here), kept for safety
            raise ApiError(
                "NON_FINITE_AFFINE",
                "quaternion (b, c, d) cannot be normalised",
                field="quatern_b",
            )
        b, c, d = b / norm, c / norm, d / norm
        a = 0.0
    else:
        a = math.sqrt(1.0 - ss)

    rot = np.array(
        [
            [a * a + b * b - c * c - d * d, 2 * b * c - 2 * a * d, 2 * b * d + 2 * a * c],
            [2 * b * c + 2 * a * d, a * a + c * c - b * b - d * d, 2 * c * d - 2 * a * b],
            [2 * b * d - 2 * a * c, 2 * c * d + 2 * a * b, a * a + d * d - c * c - b * b],
        ],
        dtype=np.float64,
    )
    scales = np.array([pixdim[1], pixdim[2], pixdim[3] * qfac], dtype=np.float64)

    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = rot * scales[np.newaxis, :]
    affine[:3, 3] = (qx, qy, qz)
    return affine


def _invert_affine(affine: np.ndarray, source: str, field: str) -> np.ndarray:
    det = float(np.linalg.det(affine[:3, :3]))
    if not math.isfinite(det) or det == 0.0:
        raise ApiError(
            "NON_INVERTIBLE_AFFINE",
            f"{source} linear part is singular (det={det}); cannot map RAS to voxels",
            field=field,
        )
    try:
        inv = np.linalg.inv(affine)
    except np.linalg.LinAlgError as exc:
        raise ApiError(
            "NON_INVERTIBLE_AFFINE",
            f"{source} affine is not invertible: {exc}",
            field=field,
        ) from exc
    if not np.all(np.isfinite(inv)):
        raise ApiError(
            "NON_INVERTIBLE_AFFINE",
            f"{source} affine inverse contains non-finite values",
            field=field,
        )
    return inv
