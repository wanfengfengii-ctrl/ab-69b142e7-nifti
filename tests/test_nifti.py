"""Unit tests for the strict NIfTI-1 parser."""
from __future__ import annotations

import math

import numpy as np
import pytest

from app.errors import ApiError
from app.nifti import DT_FLOAT32, DT_INT16, parse_nifti
from tests.niftibuild import DT_UINT8, build_nifti

SFORM_DIAG2 = [[2.0, 0.0, 0.0, 10.0],
               [0.0, 2.0, 0.0, 20.0],
               [0.0, 0.0, 2.0, 30.0]]

QFORM_BASIC = {"quat": (0.0, 0.0, 0.0), "offset": (5.0, 6.0, 7.0),
               "pixdim": (2.0, 3.0, 4.0)}


def err_code(excinfo) -> str:
    return excinfo.value.code


class TestHeaderValidation:
    def test_little_endian_int16_sform_ok(self):
        img = parse_nifti(build_nifti(endian="<", datatype=DT_INT16,
                                      dims=(4, 5, 6), sform=SFORM_DIAG2))
        assert img.endian == "<"
        assert img.datatype == DT_INT16
        assert img.shape == (4, 5, 6)
        assert img.transform_source == "sform"
        assert img.data.shape == (6, 5, 4)
        # value at voxel (i, j, k) = i + 10j + 100k
        assert img.data[5, 4, 3] == 3 + 40 + 500

    def test_big_endian_float32_qform_ok(self):
        img = parse_nifti(build_nifti(endian=">", datatype=DT_FLOAT32,
                                      dims=(3, 3, 3), qform=QFORM_BASIC))
        assert img.endian == ">"
        assert img.transform_source == "qform"
        assert img.data[1, 1, 1] == pytest.approx(111.0)

    def test_bad_sizeof_hdr(self):
        with pytest.raises(ApiError) as e:
            parse_nifti(build_nifti(sizeof_hdr=352))
        assert err_code(e) == "BAD_SIZEOF_HDR"
        assert e.value.field == "sizeof_hdr"

    def test_header_truncated(self):
        with pytest.raises(ApiError) as e:
            parse_nifti(b"\x5c\x01\x00\x00" + b"\x00" * 100)
        assert err_code(e) == "HEADER_TRUNCATED"

    def test_file_too_large(self):
        with pytest.raises(ApiError) as e:
            parse_nifti(b"\x00" * (16 * 1024 * 1024 + 1))
        assert err_code(e) == "FILE_TOO_LARGE"
        assert e.value.status == 413

    def test_bad_magic_hdr_img_pair_rejected(self):
        with pytest.raises(ApiError) as e:
            parse_nifti(build_nifti(magic=b"ni1\x00", sform=SFORM_DIAG2))
        assert err_code(e) == "BAD_MAGIC"
        assert e.value.field == "magic"

    def test_dim_must_be_3d(self):
        blob = bytearray(build_nifti(sform=SFORM_DIAG2))
        import struct
        struct.pack_into("<h", blob, 40, 4)  # dim[0] = 4
        with pytest.raises(ApiError) as e:
            parse_nifti(bytes(blob))
        assert err_code(e) == "BAD_DIM"

    def test_dim_axis_must_be_positive(self):
        blob = bytearray(build_nifti(sform=SFORM_DIAG2))
        import struct
        struct.pack_into("<h", blob, 44, 0)  # dim[2] = 0
        with pytest.raises(ApiError) as e:
            parse_nifti(bytes(blob))
        assert err_code(e) == "BAD_DIM"

    def test_unsupported_datatype(self):
        with pytest.raises(ApiError) as e:
            parse_nifti(build_nifti(datatype=DT_UINT8, sform=SFORM_DIAG2))
        assert err_code(e) == "UNSUPPORTED_DATATYPE"
        assert e.value.field == "datatype"

    def test_bitpix_mismatch(self):
        with pytest.raises(ApiError) as e:
            parse_nifti(build_nifti(datatype=DT_INT16, bitpix=32,
                                    sform=SFORM_DIAG2))
        assert err_code(e) == "BITPIX_MISMATCH"
        assert e.value.field == "bitpix"

    @pytest.mark.parametrize("vox_offset", [348.0, 351.9, 0.0, float("nan"),
                                            float("inf")])
    def test_bad_vox_offset(self, vox_offset):
        with pytest.raises(ApiError) as e:
            parse_nifti(build_nifti(vox_offset=vox_offset, sform=SFORM_DIAG2))
        assert err_code(e) == "BAD_VOX_OFFSET"
        assert e.value.field == "vox_offset"

    def test_payload_truncated(self):
        with pytest.raises(ApiError) as e:
            parse_nifti(build_nifti(sform=SFORM_DIAG2, drop=2))
        assert err_code(e) == "PAYLOAD_TRUNCATED"
        assert e.value.field == "file"

    def test_trailing_bytes(self):
        with pytest.raises(ApiError) as e:
            parse_nifti(build_nifti(sform=SFORM_DIAG2, trailing=b"\x00"))
        assert err_code(e) == "TRAILING_BYTES"
        assert e.value.field == "file"

    def test_vox_offset_with_extension_gap_ok(self):
        img = parse_nifti(build_nifti(vox_offset=360.0, sform=SFORM_DIAG2))
        assert img.data[3, 3, 3] == 333

    def test_non_finite_slope(self):
        with pytest.raises(ApiError) as e:
            parse_nifti(build_nifti(slope=float("nan"), sform=SFORM_DIAG2))
        assert err_code(e) == "NON_FINITE_SCALING"
        assert e.value.field == "scl_slope"

    def test_infinite_inter(self):
        with pytest.raises(ApiError) as e:
            parse_nifti(build_nifti(inter=float("inf"), sform=SFORM_DIAG2))
        assert err_code(e) == "NON_FINITE_SCALING"
        assert e.value.field == "scl_inter"

    def test_zero_slope_means_unscaled(self):
        img = parse_nifti(build_nifti(slope=0.0, inter=5.0, sform=SFORM_DIAG2))
        assert img.scl_slope == 1.0
        assert img.scl_inter == 0.0


class TestTransformSelection:
    def test_no_transform_rejected(self):
        with pytest.raises(ApiError) as e:
            parse_nifti(build_nifti())
        assert err_code(e) == "NO_VALID_TRANSFORM"
        assert e.value.field == "sform_code"

    def test_sform_priority_over_qform(self):
        img = parse_nifti(build_nifti(
            sform=SFORM_DIAG2, qform=QFORM_BASIC,
            sform_code=1, qform_code=2,
        ))
        assert img.transform_source == "sform"
        # sform: RAS = 2*ijk + (10, 20, 30)
        np.testing.assert_allclose(
            img.affine[:3, :],
            [[2, 0, 0, 10], [0, 2, 0, 20], [0, 0, 2, 30]],
        )

    def test_qform_used_when_no_sform(self):
        img = parse_nifti(build_nifti(qform=QFORM_BASIC))
        assert img.transform_source == "qform"
        np.testing.assert_allclose(
            img.affine[:3, :],
            [[2, 0, 0, 5], [0, 3, 0, 6], [0, 0, 4, 7]],
        )

    def test_qform_rotation_90deg_about_z(self):
        half = math.sqrt(0.5)
        qform = {"quat": (0.0, 0.0, half), "offset": (0.0, 0.0, 0.0),
                 "pixdim": (1.0, 1.0, 1.0)}
        img = parse_nifti(build_nifti(qform=qform))
        np.testing.assert_allclose(
            img.affine[:3, :3],
            [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
            atol=1e-6,
        )

    def test_qform_negative_qfac_flips_z(self):
        qform = {"quat": (0.0, 0.0, 0.0), "offset": (0.0, 0.0, 0.0),
                 "pixdim": (2.0, 2.0, 2.0), "qfac": -1.0}
        img = parse_nifti(build_nifti(qform=qform))
        np.testing.assert_allclose(
            img.affine[:3, :3],
            [[2, 0, 0], [0, 2, 0], [0, 0, -2]],
        )

    def test_non_finite_sform_row(self):
        sform = [[1.0, 0.0, 0.0, float("nan")],
                 [0.0, 1.0, 0.0, 0.0],
                 [0.0, 0.0, 1.0, 0.0]]
        with pytest.raises(ApiError) as e:
            parse_nifti(build_nifti(sform=sform))
        assert err_code(e) == "NON_FINITE_AFFINE"
        assert e.value.field == "srow_x"

    def test_non_finite_qform_quaternion(self):
        qform = {"quat": (float("nan"), 0.0, 0.0), "offset": (0, 0, 0),
                 "pixdim": (1, 1, 1)}
        with pytest.raises(ApiError) as e:
            parse_nifti(build_nifti(qform=qform))
        assert err_code(e) == "NON_FINITE_AFFINE"

    def test_singular_sform_rejected(self):
        sform = [[1.0, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 1.0, 0.0]]
        with pytest.raises(ApiError) as e:
            parse_nifti(build_nifti(sform=sform))
        assert err_code(e) == "NON_INVERTIBLE_AFFINE"
        assert e.value.field == "srow_x"

    def test_zero_pixdim_qform_rejected(self):
        qform = {"quat": (0.0, 0.0, 0.0), "offset": (0, 0, 0),
                 "pixdim": (1.0, 0.0, 1.0)}
        with pytest.raises(ApiError) as e:
            parse_nifti(build_nifti(qform=qform))
        assert err_code(e) == "NON_INVERTIBLE_AFFINE"
        assert e.value.field == "pixdim"

    def test_inverse_actually_inverts(self):
        img = parse_nifti(build_nifti(sform=SFORM_DIAG2))
        np.testing.assert_allclose(img.affine @ img.inv_affine, np.eye(4),
                                   atol=1e-12)
