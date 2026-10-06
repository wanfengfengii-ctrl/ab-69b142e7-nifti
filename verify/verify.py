"""One-shot verification service.

Runs three stages and aggregates them into a single exit code (bitmask):

* bit 0 (1) — code tests: the pytest suite (unit + API tests);
* bit 1 (2) — image build validation: the container image carries the
  expected layout, runtime and non-root user;
* bit 2 (4) — API smoke tests against the live, health-checked service,
  covering little- and big-endian files and both sform/qform affines.

Exit code 0 means every stage passed.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

EXIT_UNIT = 1
EXIT_IMAGE = 2
EXIT_SMOKE = 4

API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")

# Repository root: /app inside the image, the checkout directory locally.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SFORM = [[2.0, 0.0, 0.0, 10.0],
         [0.0, 2.0, 0.0, 20.0],
         [0.0, 0.0, 2.0, 30.0]]


def stage_unit_tests() -> list[str]:
    import pytest

    print("=== stage 1/3: code tests (pytest) ===", flush=True)
    rc = pytest.main(["-q", "-p", "no:cacheprovider", "tests"])
    return [] if rc == 0 else [f"pytest exited with code {rc}"]


def stage_image_checks() -> list[str]:
    print("=== stage 2/3: image build validation ===", flush=True)
    failures = []
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        failures.append("container process must not run as root")
    if not os.environ.get("IMAGE_VERSION"):
        failures.append("IMAGE_VERSION env is not set (Dockerfile ENV missing)")
    for rel in (
        "app/main.py",
        "app/nifti.py",
        "app/sampling.py",
        "app/errors.py",
        "requirements.txt",
    ):
        if not os.path.isfile(os.path.join(REPO_ROOT, rel)):
            failures.append(f"expected file missing from image: {rel}")
    if sys.version_info < (3, 10):
        failures.append(f"python >= 3.10 required, found {sys.version.split()[0]}")
    for module in ("numpy", "fastapi", "uvicorn", "httpx"):
        try:
            __import__(module)
        except Exception as exc:  # noqa: BLE001 - report any import failure
            failures.append(f"module {module!r} not importable: {exc}")
    return failures


def _post(client, blob: bytes, points):
    resp = client.post(
        "/api/nifti/sample",
        files={"file": ("vol.nii", blob, "application/octet-stream")},
        data={"points": json.dumps(points)},
    )
    return resp


def _ok(client, blob: bytes, points) -> dict:
    resp = _post(client, blob, points)
    _expect(resp.status_code == 200,
            f"status={resp.status_code} body={resp.text[:300]}")
    return resp.json()


def _expect(cond: bool, msg: str):
    if not cond:
        raise AssertionError(msg)


def _approx(a: float, b: float, tol: float = 1e-6) -> bool:
    return math.isclose(a, b, rel_tol=0.0, abs_tol=tol)


def smoke_checks():
    """Each check is (name, fn(client)); built lazily to defer imports."""
    from tests.niftibuild import DT_FLOAT32, DT_INT16, build_nifti
    import numpy as np

    def check_health(client):
        resp = client.get("/healthz")
        _expect(resp.status_code == 200, f"healthz -> {resp.status_code}")
        _expect(resp.json().get("status") == "ok", "healthz body mismatch")

    def check_le_int16_sform(client):
        blob = build_nifti(endian="<", datatype=DT_INT16, dims=(4, 5, 6),
                           sform=SFORM)
        points = [
            {"id": 1, "ras": [10, 20, 30]},      # voxel (0, 0, 0)
            {"id": 2, "ras": [13, 23, 35]},      # voxel (1.5, 1.5, 2.5)
            {"id": 3, "ras": [16, 28, 40]},      # voxel (3, 4, 5) corner
            {"id": 4, "ras": [100, 100, 100]},   # out of bounds
        ]
        body = _ok(client, blob, points)
        _expect(body["transform"] == "sform", f"transform={body['transform']}")
        _expect([r["id"] for r in body["results"]] == [1, 2, 3, 4],
                "request order not preserved")
        r0, r1, r2, r3 = body["results"]
        _expect(r0["status"] == "ok" and r0["voxel"] == [0.0, 0.0, 0.0]
                and _approx(r0["value"], 0.0), f"point 1: {r0}")
        _expect(r1["status"] == "ok" and r1["voxel"] == [1.5, 1.5, 2.5]
                and _approx(r1["value"], 266.5), f"point 2: {r1}")
        _expect(r2["status"] == "ok" and r2["voxel"] == [3.0, 4.0, 5.0]
                and _approx(r2["value"], 543.0), f"point 3: {r2}")
        _expect(r3["status"] == "error"
                and r3["error"]["code"] == "OUT_OF_BOUNDS", f"point 4: {r3}")

    def check_be_float32_qform(client):
        qform = {"quat": (0.0, 0.0, 0.0), "offset": (5.0, 6.0, 7.0),
                 "pixdim": (2.0, 3.0, 4.0)}
        blob = build_nifti(endian=">", datatype=DT_FLOAT32, dims=(3, 3, 3),
                           qform=qform, slope=0.5, inter=2.0)
        points = [{"id": 10, "ras": [7, 9, 11]},   # voxel (1, 1, 1)
                  {"id": 11, "ras": [5, 6, 7]}]    # voxel (0, 0, 0)
        body = _ok(client, blob, points)
        _expect(body["transform"] == "qform", f"transform={body['transform']}")
        r0, r1 = body["results"]
        _expect(r0["status"] == "ok" and r0["voxel"] == [1.0, 1.0, 1.0]
                and _approx(r0["value"], 0.5 * 111 + 2), f"point 10: {r0}")
        _expect(r1["status"] == "ok" and _approx(r1["value"], 2.0),
                f"point 11: {r1}")

    def check_sform_priority(client):
        # Both transforms present: sform (offset 100/200/300) must win over
        # the qform (identity at origin).
        qform = {"quat": (0.0, 0.0, 0.0), "offset": (0.0, 0.0, 0.0),
                 "pixdim": (1.0, 1.0, 1.0)}
        sform = [[1.0, 0.0, 0.0, 100.0],
                 [0.0, 1.0, 0.0, 200.0],
                 [0.0, 0.0, 1.0, 300.0]]
        blob = build_nifti(endian=">", datatype=DT_INT16, dims=(2, 2, 2),
                           sform=sform, qform=qform,
                           sform_code=1, qform_code=2)
        body = _ok(client, blob, [{"id": 1, "ras": [100, 200, 300]}])
        _expect(body["transform"] == "sform", f"transform={body['transform']}")
        r = body["results"][0]
        _expect(r["status"] == "ok" and r["voxel"] == [0.0, 0.0, 0.0],
                f"sform priority not honoured: {r}")

    def check_qform_qfac_and_rotation(client):
        # 90-degree rotation about z with qfac=-1: exercises quaternion
        # reconstruction and the pixdim[0] handedness flag.
        half = math.sqrt(0.5)
        qform = {"quat": (0.0, 0.0, half), "offset": (0.0, 0.0, 0.0),
                 "pixdim": (2.0, 2.0, 2.0), "qfac": -1.0}
        blob = build_nifti(endian="<", datatype=DT_FLOAT32, dims=(3, 3, 3),
                           qform=qform)
        # affine: RAS = ( -2j, 2i, -2k ); voxel (1, 0, 1) -> RAS (0, 2, -2)
        body = _ok(client, blob, [{"id": 1, "ras": [0, 2, -2]}])
        r = body["results"][0]
        _expect(r["status"] == "ok", f"unexpected error: {r}")
        _expect(all(_approx(a, b, 1e-5) for a, b in
                    zip(r["voxel"], [1.0, 0.0, 1.0])), f"voxel: {r}")
        _expect(_approx(r["value"], 101.0, 1e-4), f"value: {r}")

    def check_trailing_bytes_rejected(client):
        blob = build_nifti(endian="<", datatype=DT_INT16, dims=(2, 2, 2),
                           sform=SFORM, trailing=b"junk")
        resp = _post(client, blob, [{"id": 1, "ras": [0, 0, 0]}])
        _expect(resp.status_code == 400, f"status={resp.status_code}")
        err = resp.json()["error"]
        _expect(err["code"] == "TRAILING_BYTES" and err["field"] == "file",
                f"error={err}")

    def check_no_transform_rejected(client):
        blob = build_nifti(endian=">", datatype=DT_FLOAT32, dims=(2, 2, 2))
        resp = _post(client, blob, [{"id": 1, "ras": [0, 0, 0]}])
        err = resp.json()["error"]
        _expect(resp.status_code == 400 and err["code"] == "NO_VALID_TRANSFORM",
                f"error={err}")

    def check_non_finite_data_per_point(client):
        data = np.zeros((4, 4, 4), dtype=np.float32)
        data[1, 1, 1] = np.nan  # voxel (1, 1, 1)
        blob = build_nifti(endian="<", datatype=DT_FLOAT32, dims=(4, 4, 4),
                           sform=SFORM, data=data)
        points = [{"id": 5, "ras": [12, 22, 32]},   # voxel (1,1,1) -> NaN
                  {"id": 6, "ras": [16, 26, 36]}]   # voxel (3,3,3) -> fine
        body = _ok(client, blob, points)
        r0, r1 = body["results"]
        _expect(r0["status"] == "error"
                and r0["error"]["code"] == "NON_FINITE_DATA"
                and r0["id"] == 5, f"point 5: {r0}")
        _expect(r1["status"] == "ok" and _approx(r1["value"], 0.0),
                f"point 6: {r1}")

    def check_request_validation(client):
        blob = build_nifti(endian="<", datatype=DT_INT16, dims=(2, 2, 2),
                           sform=SFORM)
        dup = _post(client, blob, [{"id": 1, "ras": [0, 0, 0]},
                                   {"id": 1, "ras": [1, 1, 1]}])
        _expect(dup.status_code == 400
                and dup.json()["error"]["code"] == "POINT_ID_DUPLICATE",
                f"duplicate ids: {dup.status_code} {dup.text}")
        many = _post(client, blob,
                     [{"id": i, "ras": [0, 0, 0]} for i in range(257)])
        _expect(many.status_code == 400
                and many.json()["error"]["code"] == "POINTS_COUNT",
                f"257 points: {many.status_code} {many.text}")

    return [
        ("health endpoint", check_health),
        ("little-endian int16 + sform sampling", check_le_int16_sform),
        ("big-endian float32 + qform sampling", check_be_float32_qform),
        ("sform priority over qform", check_sform_priority),
        ("qform rotation + negative qfac", check_qform_qfac_and_rotation),
        ("trailing bytes rejected", check_trailing_bytes_rejected),
        ("missing transforms rejected", check_no_transform_rejected),
        ("non-finite data localised per point", check_non_finite_data_per_point),
        ("request validation (duplicate ids, >256 points)", check_request_validation),
    ]


def stage_smoke() -> list[str]:
    print("=== stage 3/3: API smoke tests ===", flush=True)
    import httpx

    failures = []
    with httpx.Client(base_url=API_BASE_URL, timeout=15.0) as client:
        deadline = time.monotonic() + 60.0
        while True:
            try:
                if client.get("/healthz").status_code == 200:
                    break
            except Exception:  # noqa: BLE001 - service may still be starting
                pass
            if time.monotonic() > deadline:
                return [f"service at {API_BASE_URL} did not become healthy in 60s"]
            time.sleep(1.0)

        for name, check in smoke_checks():
            try:
                check(client)
            except Exception as exc:  # noqa: BLE001 - aggregate all failures
                failures.append(f"{name}: {exc}")
                print(f"  FAIL {name}: {exc}", flush=True)
            else:
                print(f"  PASS {name}", flush=True)
    return failures


def _guard(stage: str, fn):
    """Run a stage, converting unexpected exceptions into stage failures."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - a crashing stage is a failure
        return [f"{stage} stage crashed: {exc!r}"]


def main() -> int:
    exit_code = 0

    unit_failures = _guard("unit", stage_unit_tests)
    if unit_failures:
        exit_code |= EXIT_UNIT

    image_failures = _guard("image", stage_image_checks)
    if image_failures:
        exit_code |= EXIT_IMAGE

    smoke_failures = _guard("smoke", stage_smoke)
    if smoke_failures:
        exit_code |= EXIT_SMOKE

    print("=== verification summary ===", flush=True)
    for stage, failures in (
        ("code tests", unit_failures),
        ("image build validation", image_failures),
        ("API smoke", smoke_failures),
    ):
        status = "PASS" if not failures else "FAIL"
        print(f"  {stage}: {status}", flush=True)
        for failure in failures:
            print(f"    - {failure}", flush=True)
    print(f"verify exit code: {exit_code}", flush=True)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
