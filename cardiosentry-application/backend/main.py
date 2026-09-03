"""CardioSentry — FastAPI entrypoint.

Startup sequence (plan §7.4):
  1. settings,  2. descriptor discovery + validation,  3. build active model,
  4. warm-up forward (in runner.load_model),  5. verify_model self-check when a
  fixture is configured,  6. LAN URLs + terminal QR code,  7. static mount.
A broken model refuses startup — never serve with one (plan §16 #10).
"""
from __future__ import annotations

import json
import logging
import socket
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.api import api_router
from backend.api.state import AppState, create_state
from backend.errors import ApiError
from backend.settings import APP_ROOT, settings

log = logging.getLogger("cardiosentry")


def _git_sha() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5,
                             cwd=APP_ROOT.parent)
        return out.stdout.strip() or None
    except Exception:
        return None


def _lan_addresses() -> list[str]:
    addrs: set[str] = set()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(1.0)                # never block startup on the network
        s.connect(("8.8.8.8", 80))
        addrs.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127."):
                addrs.add(ip)
    except OSError:
        pass
    return sorted(addrs)


def _print_access_urls() -> None:
    host = settings.bind_host
    if host == "127.0.0.1":
        print(f"\n  CardioSentry running LOCAL-ONLY at http://127.0.0.1:{settings.port}\n")
        return
    print(f"\n  CardioSentry is serving on port {settings.port}.")
    urls = [f"http://{ip}:{settings.port}" for ip in _lan_addresses()]
    for u in urls:
        print(f"    {u}")
    if not urls:
        print("    (could not determine LAN IPs)")
    primary = urls[0] if urls else f"http://127.0.0.1:{settings.port}"
    try:
        import qrcode
        qr = qrcode.QRCode(border=2)
        qr.add_data(primary)
        qr.make()
        qr.print_ascii(invert=True)
    except Exception:
        pass
    print(f"\n  Open this on the phone (same Wi-Fi):  {primary}\n")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    log.info("starting CardioSentry…")
    st: AppState = create_state()
    app.state.csd = st
    app.state.git_sha = _git_sha()
    log.info("active model: %s (views=%s, %d threads)",
             st.active_model.id, st.active_model.views, st.active_model.num_threads)
    if settings.verify_fixture_image:
        _startup_self_check(st)
    from backend.services.storage import free_space_gb
    free = free_space_gb()
    if free < settings.min_free_gb_warn:
        log.warning("low disk space: %.1f GB free in %s (warn below %.0f GB)",
                    free, settings.data_dir, settings.min_free_gb_warn)
    _print_access_urls()
    yield
    log.info("shutdown complete")


def _startup_self_check(st: AppState) -> None:
    """verify_model at startup (warn-only, plan §10.6 / §7.4 step 5)."""
    try:
        from scripts.verify_model import verify_model
        fixture = Path(settings.verify_fixture_image)
        if not fixture.is_file():
            log.warning("verify fixture %s not found — skipping self-check", fixture)
            return
        # The active descriptor's `verify` block wins: the reference vector and
        # the tolerance it may pass are properties of the checkpoint, so a
        # model swap must not leave this comparing against another run's
        # numbers (or against another run's operating point on the sigmoid).
        ld = st.descriptors[st.active_model.id]
        vspec = ld.descriptor.verify
        expected_path = (APP_ROOT / vspec.expected_scores if vspec.expected_scores
                         else Path(settings.verify_expected_scores_json)
                         if settings.verify_expected_scores_json else None)
        expected = None
        if expected_path and expected_path.is_file():
            expected = json.loads(expected_path.read_text(encoding="utf-8"))
        elif expected_path:
            log.warning("expected-scores file %s not found — shape check only",
                        expected_path)
        report = verify_model(
            ld, fixture, expected,
            vspec.score_tolerance or settings.verify_score_tolerance)
        log.info("verify_model self-check passed: max |Δ| = %s",
                 report.get("max_delta", "n/a"))
    except Exception as exc:
        log.warning("verify_model self-check FAILED (serving anyway): %s", exc)


app = FastAPI(title="CardioSentry", version="0.1.0", lifespan=lifespan)


@app.middleware("http")
async def token_gate(request: Request, call_next):
    """Optional shared-secret gate (§19.2 / §19.3): APP_TOKEN."""
    if settings.app_token and request.url.path.startswith("/api"):
        auth = request.headers.get("authorization", "")
        token = request.query_params.get("token", "")
        supplied = auth.removeprefix("Bearer ").strip() or token
        if supplied != settings.app_token:
            return JSONResponse(status_code=401,
                                content={"code": "UNAUTHORIZED",
                                         "message": "invalid token"})
    return await call_next(request)


@app.exception_handler(ApiError)
async def api_error_handler(request: Request, exc: ApiError):
    return JSONResponse(status_code=exc.status_code,
                        content={"code": exc.code, "message": exc.message})


@app.exception_handler(Exception)
async def unhandled_handler(request: Request, exc: Exception):
    log.exception("unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500,
                        content={"code": "INTERNAL_ERROR",
                                 "message": "Unexpected server error."})


app.include_router(api_router)

# Static frontend (same origin — no CORS anywhere, §15.2). Mounted last so
# /api always wins. dist may not exist yet (pure-API dev mode): then serve a
# tiny placeholder at /.
_dist = Path(settings.frontend_dist)
if _dist.is_dir():
    app.mount("/", StaticFiles(directory=str(_dist), html=True), name="frontend")
else:
    @app.get("/")
    def _index():
        return JSONResponse({
            "code": "FRONTEND_NOT_BUILT",
            "message": "frontend/dist not found — run `cd frontend && npm install && npm run build`",
        })


def main() -> int:
    import uvicorn
    uvicorn.run("backend.main:app", host=settings.bind_host, port=settings.port,
                workers=1, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
