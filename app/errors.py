"""Stable error codes shared by the API layer.

Two error channels exist:

* :class:`ApiError` — request/file level problems. Rendered as an HTTP 4xx
  JSON body ``{"error": {"code", "message", "field", ...}}`` where ``field``
  names the offending NIfTI header field or form field.
* :class:`PointError` — per-point problems (out of bounds, non-finite data).
  Reported inside a 200 response, in the offending point's result entry, so
  the point ``id`` always localises it.
"""
from __future__ import annotations


class ApiError(Exception):
    """Request-level error with a stable machine-readable code."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        field: str | None = None,
        status: int = 400,
        details: dict | None = None,
    ):
        super().__init__(message)
        self.code = code
        self.message = message
        self.field = field
        self.status = status
        self.details = details or {}

    def to_dict(self) -> dict:
        err: dict = {"code": self.code, "message": self.message}
        if self.field is not None:
            err["field"] = self.field
        if self.details:
            err["details"] = self.details
        return {"error": err}


class PointError(Exception):
    """Error attached to a single sampling point (identified by its id)."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message
