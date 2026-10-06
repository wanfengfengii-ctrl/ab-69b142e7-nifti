"""Unit tests for RAS->voxel mapping and trilinear interpolation."""
from __future__ import annotations

import numpy as np
import pytest

from app.errors import PointError
from app.nifti import DT_FLOAT32, DT_INT16, parse_nifti
from app.sampling import sample_point
from tests.niftibuild import build_nifti

# RAS = 2*ijk + (10, 20, 30); volume value at (i, j, k) is i + 10j + 100k.
SFORM = [[2.0, 0.0, 0.0, 10.0],
         [0.0, 2.0, 0.0, 20.0],
         [0.0, 0.0, 2.0, 30.0]]
DIMS = (4, 5, 6)


def make_img(**kwargs):
    kwargs.setdefault("sform", SFORM)
    kwargs.setdefault("dims", DIMS)
    return parse_nifti(build_nifti(**kwargs))


def ras(i, j, k):
    return [10.0 + 2.0 * i, 20.0 + 2.0 * j, 30.0 + 2.0 * k]


class TestTrilinear:
    def test_exact_grid_point(self):
        voxel, value = sample_point(make_img(), ras(1, 2, 3))
        assert voxel == [1.0, 2.0, 3.0]
        assert value == pytest.approx(1 + 20 + 300)

    def test_centroid_interpolates_linear_pattern(self):
        voxel, value = sample_point(make_img(), ras(1.5, 2.5, 3.5))
        assert voxel == [1.5, 2.5, 3.5]
        assert value == pytest.approx(1.5 + 25 + 350)

    def test_quarter_offset(self):
        _, value = sample_point(make_img(), ras(0.25, 0.5, 0.75))
        assert value == pytest.approx(0.25 + 5 + 75)

    def test_upper_boundary_corner_uses_endpoint(self):
        voxel, value = sample_point(make_img(), ras(3, 4, 5))
        assert voxel == [3.0, 4.0, 5.0]
        assert value == pytest.approx(3 + 40 + 500)

    def test_origin_corner(self):
        voxel, value = sample_point(make_img(), ras(0, 0, 0))
        assert voxel == [0.0, 0.0, 0.0]
        assert value == pytest.approx(0.0)

    def test_single_slice_axis_pinned(self):
        img = make_img(dims=(1, 3, 3))
        voxel, value = sample_point(img, ras(0, 1.5, 1.5))
        assert voxel == [0.0, 1.5, 1.5]
        assert value == pytest.approx(0 + 15 + 150)

    def test_scaling_applied_before_interpolation(self):
        img = make_img(slope=2.0, inter=3.0)
        _, value = sample_point(img, ras(1.5, 2.5, 3.5))
        assert value == pytest.approx(2.0 * (1.5 + 25 + 350) + 3.0)

    def test_float32_big_endian_values(self):
        img = make_img(endian=">", datatype=DT_FLOAT32)
        _, value = sample_point(img, ras(2, 3, 4))
        assert value == pytest.approx(2 + 30 + 400)


class TestDomain:
    @pytest.mark.parametrize(
        "point",
        [ras(-0.1, 0, 0), ras(0, -1, 0), ras(0, 0, 5.01), ras(3.01, 0, 0),
         ras(0, 4.2, 0), ras(100, 100, 100)],
    )
    def test_out_of_bounds(self, point):
        with pytest.raises(PointError) as e:
            sample_point(make_img(), point)
        assert e.value.code == "OUT_OF_BOUNDS"

    def test_boundary_tolerance_accepts_float_noise(self):
        # A hair outside the closed domain is float noise, not a violation.
        img = make_img()
        voxel, _ = sample_point(img, [10.0 - 2e-7, 20.0, 30.0])
        assert voxel[0] == 0.0

    def test_non_finite_data_in_neighbourhood(self):
        data = np.zeros((6, 5, 4), dtype=np.float32)
        data[2, 2, 2] = np.nan  # voxel (i=2, j=2, k=2)
        img = make_img(datatype=DT_FLOAT32, data=data)
        with pytest.raises(PointError) as e:
            sample_point(img, ras(2, 2, 2))
        assert e.value.code == "NON_FINITE_DATA"

    def test_non_finite_data_far_away_is_fine(self):
        data = np.zeros((6, 5, 4), dtype=np.float32)
        data[0, 0, 0] = np.inf  # voxel (0, 0, 0)
        img = make_img(datatype=DT_FLOAT32, data=data)
        _, value = sample_point(img, ras(3, 4, 5))
        assert value == 0.0


class TestQformSampling:
    def test_qform_round_trip(self):
        qform = {"quat": (0.0, 0.0, 0.0), "offset": (5.0, 6.0, 7.0),
                 "pixdim": (2.0, 3.0, 4.0)}
        img = parse_nifti(build_nifti(endian=">", datatype=DT_FLOAT32,
                                      dims=(3, 3, 3), qform=qform))
        voxel, value = sample_point(img, [5 + 2, 6 + 3, 7 + 4])
        assert voxel == [1.0, 1.0, 1.0]
        assert value == pytest.approx(111.0)
