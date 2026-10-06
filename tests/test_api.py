"""API-level tests via FastAPI's TestClient."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.nifti import MAX_FILE_BYTES
from tests.niftibuild import DT_FLOAT32, build_nifti

client = TestClient(app)

SFORM = [[2.0, 0.0, 0.0, 10.0],
         [0.0, 2.0, 0.0, 20.0],
         [0.0, 0.0, 2.0, 30.0]]


def post(file_bytes: bytes | None, points: object, **kwargs):
    files = None
    if file_bytes is not None:
        files = {"file": ("vol.nii", file_bytes, "application/octet-stream")}
    data = {}
    if points is not None:
        data["points"] = points if isinstance(points, str) else json.dumps(points)
    return client.post("/api/nifti/sample", files=files, data=data, **kwargs)


def good_file(**kwargs):
    kwargs.setdefault("sform", SFORM)
    kwargs.setdefault("dims", (4, 5, 6))
    return build_nifti(**kwargs)


class TestHappyPath:
    def test_results_keep_request_order(self):
        points = [
            {"id": 7, "ras": [10, 20, 30]},        # voxel (0, 0, 0)
            {"id": 3, "ras": [13, 23, 35]},        # voxel (1.5, 1.5, 2.5)
            {"id": 9, "ras": [16, 28, 40]},        # voxel (3, 4, 5)
        ]
        resp = post(good_file(), points)
        assert resp.status_code == 200
        body = resp.json()
        assert body["transform"] == "sform"
        assert [r["id"] for r in body["results"]] == [7, 3, 9]
        r0, r1, r2 = body["results"]
        assert r0["voxel"] == [0.0, 0.0, 0.0]
        assert r0["value"] == pytest.approx(0.0)
        assert r1["voxel"] == [1.5, 1.5, 2.5]
        assert r1["value"] == pytest.approx(1.5 + 15 + 250)
        assert r2["voxel"] == [3.0, 4.0, 5.0]
        assert r2["value"] == pytest.approx(3 + 40 + 500)

    def test_mixed_ok_and_error_points(self):
        points = [
            {"id": 1, "ras": [10, 20, 30]},
            {"id": 2, "ras": [1000, 0, 0]},
        ]
        body = post(good_file(), points).json()
        assert body["results"][0]["status"] == "ok"
        err = body["results"][1]
        assert err["status"] == "error"
        assert err["error"]["code"] == "OUT_OF_BOUNDS"
        assert err["id"] == 2

    def test_qform_transform_source_reported(self):
        qform = {"quat": (0.0, 0.0, 0.0), "offset": (0.0, 0.0, 0.0),
                 "pixdim": (1.0, 1.0, 1.0)}
        blob = build_nifti(endian=">", datatype=DT_FLOAT32, dims=(3, 3, 3),
                           qform=qform)
        body = post(blob, [{"id": 0, "ras": [1, 1, 1]}]).json()
        assert body["transform"] == "qform"
        assert body["results"][0]["value"] == pytest.approx(111.0)


class TestRequestValidation:
    def test_healthz(self):
        resp = client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_wrong_content_type(self):
        resp = client.post("/api/nifti/sample", json={"points": []})
        assert resp.status_code == 415
        assert resp.json()["error"]["code"] == "UNSUPPORTED_MEDIA_TYPE"

    def test_missing_file(self):
        # Multipart form that carries no 'file' field at all.
        resp = client.post(
            "/api/nifti/sample",
            files={"other": ("x.txt", b"hi", "text/plain")},
            data={"points": json.dumps([{"id": 1, "ras": [0, 0, 0]}])},
        )
        assert resp.status_code == 400
        err = resp.json()["error"]
        assert err["code"] == "FILE_MISSING"
        assert err["field"] == "file"

    def test_multiple_files(self):
        blob = good_file()
        resp = client.post(
            "/api/nifti/sample",
            files=[("file", ("a.nii", blob)), ("file", ("b.nii", blob))],
            data={"points": json.dumps([{"id": 1, "ras": [0, 0, 0]}])},
        )
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "MULTIPLE_FILES"

    def test_file_too_large(self):
        resp = post(b"\x00" * (MAX_FILE_BYTES + 1),
                    [{"id": 1, "ras": [0, 0, 0]}])
        assert resp.status_code == 413
        assert resp.json()["error"]["code"] == "FILE_TOO_LARGE"

    def test_missing_points(self):
        resp = post(good_file(), None)
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "POINTS_MISSING"

    def test_points_invalid_json(self):
        resp = post(good_file(), "{not json")
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "POINTS_INVALID_JSON"

    def test_points_not_an_array(self):
        resp = post(good_file(), {"id": 1})
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "POINTS_INVALID"

    def test_points_empty(self):
        resp = post(good_file(), [])
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "POINTS_COUNT"

    def test_points_too_many(self):
        pts = [{"id": i, "ras": [0, 0, 0]} for i in range(257)]
        resp = post(good_file(), pts)
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "POINTS_COUNT"

    def test_points_max_accepted(self):
        pts = [{"id": i, "ras": [10, 20, 30]} for i in range(256)]
        resp = post(good_file(), pts)
        assert resp.status_code == 200
        assert len(resp.json()["results"]) == 256

    def test_duplicate_point_id(self):
        pts = [{"id": 5, "ras": [0, 0, 0]}, {"id": 5, "ras": [1, 1, 1]}]
        resp = post(good_file(), pts)
        assert resp.status_code == 400
        err = resp.json()["error"]
        assert err["code"] == "POINT_ID_DUPLICATE"
        assert "5" in err["message"]

    @pytest.mark.parametrize("bad_id", [1.5, "1", True, None, [1]])
    def test_point_id_must_be_int(self, bad_id):
        resp = post(good_file(), [{"id": bad_id, "ras": [0, 0, 0]}])
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "POINT_ID_INVALID"

    def test_ras_wrong_length(self):
        resp = post(good_file(), [{"id": 1, "ras": [1, 2]}])
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "POINT_RAS_INVALID"

    def test_ras_non_number(self):
        resp = post(good_file(), [{"id": 1, "ras": [1, "x", 3]}])
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "POINT_RAS_INVALID"

    def test_ras_non_finite(self):
        # Python's json accepts NaN/Infinity literals; the API must not.
        resp = post(good_file(), '[{"id": 1, "ras": [NaN, 0, 0]}]')
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "POINT_COORD_NON_FINITE"


class TestStructuralErrors:
    def test_trailing_bytes_error_body(self):
        resp = post(good_file(trailing=b"junk"),
                    [{"id": 1, "ras": [0, 0, 0]}])
        assert resp.status_code == 400
        err = resp.json()["error"]
        assert err["code"] == "TRAILING_BYTES"
        assert err["field"] == "file"
        assert err["details"]["actual_bytes"] - err["details"]["expected_bytes"] == 4

    def test_no_transform_error_body(self):
        resp = post(build_nifti(dims=(2, 2, 2)),
                    [{"id": 1, "ras": [0, 0, 0]}])
        assert resp.status_code == 400
        err = resp.json()["error"]
        assert err["code"] == "NO_VALID_TRANSFORM"
        assert err["field"] == "sform_code"

    def test_singular_affine_error_body(self):
        sform = [[1.0, 0.0, 0.0, 0.0],
                 [0.0, 1.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0, 0.0]]
        resp = post(build_nifti(sform=sform), [{"id": 1, "ras": [0, 0, 0]}])
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "NON_INVERTIBLE_AFFINE"

    def test_non_finite_data_per_point(self):
        import numpy as np
        data = np.zeros((6, 5, 4), dtype=np.float32)
        data[0, 0, 0] = np.nan
        blob = good_file(datatype=DT_FLOAT32, data=data)
        body = post(blob, [{"id": 42, "ras": [10, 20, 30]},
                           {"id": 43, "ras": [16, 28, 40]}]).json()
        assert body["results"][0]["status"] == "error"
        assert body["results"][0]["error"]["code"] == "NON_FINITE_DATA"
        assert body["results"][0]["id"] == 42
        assert body["results"][1]["status"] == "ok"
