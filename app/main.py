"""HTTP API for the NIfTI RAS sampling service.

Endpoints
---------
GET  /healthz             readiness probe
POST /api/nifti/sample    multipart form: one ``file`` field (<= 16 MiB .nii)
                          plus a ``points`` field with a JSON array of
                          ``{"id": <int>, "ras": [x, y, z]}`` (1..256 items,
                          unique ids).  Results keep the request order.
"""
from __future__ import annotations

import json
import math

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.datastructures import UploadFile

from .errors import ApiError, PointError
from .nifti import MAX_FILE_BYTES, parse_nifti
from .sampling import sample_point

MAX_POINTS = 256
# Multipart overhead (boundaries, headers, the points JSON) on top of the
# 16 MiB file allowance; requests larger than this are refused up front.
MAX_REQUEST_BYTES = MAX_FILE_BYTES + 1024 * 1024

app = FastAPI(title="nifti-qc-sampler", version="1.0.0")


@app.exception_handler(ApiError)
async def api_error_handler(_request: Request, exc: ApiError) -> JSONResponse:
    return JSONResponse(status_code=exc.status, content=exc.to_dict())


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@app.get("/")
async def root() -> dict:
    return {
        "service": "nifti-qc-sampler",
        "version": "1.0.0",
        "endpoints": {"sample": "/api/nifti/sample", "health": "/healthz"},
    }


@app.post("/api/nifti/sample")
async def sample(request: Request) -> dict:
    content_type = request.headers.get("content-type", "")
    if not content_type.lower().startswith("multipart/form-data"):
        raise ApiError(
            "UNSUPPORTED_MEDIA_TYPE",
            "content-type must be multipart/form-data",
            status=415,
        )
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > MAX_REQUEST_BYTES:
                raise ApiError(
                    "REQUEST_TOO_LARGE",
                    f"request body exceeds {MAX_REQUEST_BYTES} bytes",
                    status=413,
                )
        except ValueError:
            pass  # unparsable Content-Length: let the form parser decide

    form = await request.form()
    files = [v for v in form.getlist("file") if isinstance(v, UploadFile)]
    if not files:
        raise ApiError(
            "FILE_MISSING",
            "multipart form must carry exactly one NIfTI file in field 'file'",
            field="file",
        )
    if len(files) > 1:
        raise ApiError(
            "MULTIPLE_FILES",
            f"exactly one file is accepted; got {len(files)}",
            field="file",
        )
    content = await files[0].read()
    img = parse_nifti(content)  # raises ApiError on any structural problem

    points = _parse_points(form.get("points"))

    results = []
    for pid, ras in points:
        try:
            voxel, value = sample_point(img, ras)
            results.append(
                {"id": pid, "status": "ok", "voxel": voxel, "value": value}
            )
        except PointError as exc:
            results.append(
                {
                    "id": pid,
                    "status": "error",
                    "error": {"code": exc.code, "message": exc.message},
                }
            )
    return {"transform": img.transform_source, "results": results}


def _parse_points(raw: object) -> list[tuple[int, list[float]]]:
    if raw is None or not isinstance(raw, str) or not raw.strip():
        raise ApiError(
            "POINTS_MISSING",
            "form field 'points' must be a JSON array of "
            '{"id": <int>, "ras": [x, y, z]}',
            field="points",
        )
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ApiError(
            "POINTS_INVALID_JSON",
            f"'points' is not valid JSON: {exc}",
            field="points",
        ) from exc
    if not isinstance(obj, list):
        raise ApiError(
            "POINTS_INVALID",
            "'points' must be a JSON array of point objects",
            field="points",
        )
    if not 1 <= len(obj) <= MAX_POINTS:
        raise ApiError(
            "POINTS_COUNT",
            f"'points' must contain between 1 and {MAX_POINTS} entries; got {len(obj)}",
            field="points",
        )

    points: list[tuple[int, list[float]]] = []
    seen: set[int] = set()
    for index, item in enumerate(obj):
        where = f"points[{index}]"
        if not isinstance(item, dict):
            raise ApiError(
                "POINT_INVALID",
                f"{where} must be an object with 'id' and 'ras'",
                field="points",
            )
        pid = item.get("id")
        if not isinstance(pid, int) or isinstance(pid, bool):
            raise ApiError(
                "POINT_ID_INVALID",
                f"{where}.id must be an integer; got {pid!r}",
                field="points",
            )
        if pid in seen:
            raise ApiError(
                "POINT_ID_DUPLICATE",
                f"duplicate point id {pid} (at {where})",
                field="points",
            )
        seen.add(pid)

        ras = item.get("ras")
        if not isinstance(ras, (list, tuple)) or len(ras) != 3:
            raise ApiError(
                "POINT_RAS_INVALID",
                f"{where}.ras must be an array of exactly 3 finite numbers",
                field="points",
            )
        coords = []
        for component in ras:
            if isinstance(component, bool) or not isinstance(component, (int, float)):
                raise ApiError(
                    "POINT_RAS_INVALID",
                    f"{where}.ras entries must be numbers; got {component!r}",
                    field="points",
                )
            if not math.isfinite(component):
                raise ApiError(
                    "POINT_COORD_NON_FINITE",
                    f"{where}.ras entries must be finite; got {component!r}",
                    field="points",
                )
            coords.append(float(component))
        points.append((pid, coords))
    return points
