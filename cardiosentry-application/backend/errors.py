"""Typed API errors: every error response carries a stable `code` (plan §16)."""
from __future__ import annotations


class ApiError(Exception):
    def __init__(self, status_code: int, code: str, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


class NotFound(ApiError):
    def __init__(self, code: str = "NOT_FOUND", message: str = "Not found"):
        super().__init__(404, code, message)
