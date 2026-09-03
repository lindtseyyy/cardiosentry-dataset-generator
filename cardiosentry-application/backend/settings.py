"""Application settings.

Everything is overridable via environment variables (or a `.env` file next to
the app). Paths default relative to the repository root of this checkout so the
app runs unconfigured from a fresh clone.
"""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

APP_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CARDIOSENTRY_", env_file=APP_ROOT / ".env", extra="ignore"
    )

    # ---------------------------------------------------------------- serving
    host: str = "0.0.0.0"          # LAN-serving default; set 127.0.0.1 for --local-only
    port: int = 8000
    local_only: bool = False       # overrides host to 127.0.0.1
    app_token: str | None = None   # optional shared-secret gate (§19.2); unused = open

    # ---------------------------------------------------------------- paths
    data_dir: Path = APP_ROOT / "captures"
    models_dir: Path = APP_ROOT / "models"
    frontend_dist: Path = APP_ROOT / "frontend" / "dist"
    active_model: str | None = None   # model id; None = follow models/active.yaml symlink

    # ---------------------------------------------------------------- model
    device: str = "cpu"
    dtype: str = "float32"
    num_threads: int = 4
    # Startup self-check (§10.6): the fixture + the prototype's stored scores
    # for it are checked in, so this is on by default — it catches timm/weights
    # drift at every boot. Set the fixture to null to disable.
    verify_fixture_image: str | None = str(APP_ROOT / "tests" / "fixtures" / "corpus_sheet.jpg")
    verify_expected_scores_json: str | None = str(
        APP_ROOT / "tests" / "fixtures" / "expected_scores.json")
    verify_score_tolerance: float = 2e-3      # CPU-fp32 vs training GPU-AMP gap (§17.9)
    # Both of the above are FALLBACKS: a descriptor's `verify` block wins, so
    # the active model is always checked against its own run's stored vector.

    # ---------------------------------------------------------------- ingest limits
    max_upload_bytes: int = 30 * 1024 * 1024   # 30 MB
    max_image_pixels: int = 200_000_000        # decompression-bomb guard
    strip_gps: bool = True                     # §13.4: GPS EXIF removed from stored bytes

    # ---------------------------------------------------------------- detection
    detect_downscale_to: int = 1024
    detect_min_area_frac: float = 0.15
    detect_expected_aspect: float | None = 1.294  # US Letter landscape; null disables prior
    detect_confidence_warn: float = 0.55
    paper_width_mm: float = 279.4               # US Letter landscape, for px/mm metric

    # ---------------------------------------------------------------- rectification
    geometry_mode: str = "training_canvas"      # training_canvas | native_aspect
    canvas_px: list[int] = [1650, 1275]         # pre-margin canonical canvas (F5)
    margin_px: int = 18                         # paper-coloured border

    # ---------------------------------------------------------------- preprocessing
    # Fallback only — the active descriptor's value wins. [height, width], the
    # geometry the round-3 nat768 checkpoints were trained at.
    input_size: list[int] = [768, 1024]
    resample: str = "pil_bicubic"
    match_training_jpeg: bool = True            # §9.3.3: in-memory q95 round-trip
    jpeg_quality: int = 95
    normalize_mean: list[float] = [0.485, 0.456, 0.406]
    normalize_std: list[float] = [0.229, 0.224, 0.225]

    # ---------------------------------------------------------------- quality (§14)
    quality_block_on_poor: bool = False         # warn, never block
    warn_blur_varlap: float = 100.0             # computed on corrected.png (fixed canvas)
    warn_brightness_lo: float = 0.25
    warn_brightness_hi: float = 0.85
    warn_contrast_p5p95: float = 0.30
    warn_clipped_high_frac: float = 0.05
    warn_clipped_low_frac: float = 0.05
    warn_glare_blob_frac: float = 0.01
    warn_px_per_mm: float = 6.0                 # training render = 5.91 px/mm
    warn_skew_deg: float = 25.0
    warn_area_fraction: float = 0.35

    # ---------------------------------------------------------------- explainability (§XAI)
    # Every method here costs whole model passes on CPU, so the defaults are
    # the ones that stay interactive on the serving laptop rather than the
    # ones a paper would use. Raise ig_steps for a figure; watch the
    # `completeness_delta` the run reports to know when it was enough.
    explain_enabled: bool = True
    explain_ig_steps: int = 16                  # Riemann midpoint steps
    explain_ig_max_steps: int = 256
    explain_ig_baseline: str = "white"          # white | black | grey | blur
    # Gradient checkpointing for the two input-gradient methods: recompute
    # activations in the backward instead of storing them. Measured on
    # efficientnetv2_s_cbam@768x1024, the Jacobian goes +2478 MB / 7.6 s ->
    # +506 MB / 14.5 s. `auto` turns it on only when the machine could not
    # otherwise fit the run — which is what stops a 2.6 GB-free laptop from
    # being OOM-killed mid-request. Grad-CAM never needs it (it records no
    # graph before its target layer).
    explain_memory_policy: str = "auto"          # auto | always | never
    explain_overlay_alpha: float = 0.5
    explain_colormap: str = "turbo"             # positive maps; signed maps use bwr
    explain_percentile_clip: float = 99.0       # robust max for map normalization
    explain_save_raw: bool = False              # also write raw.npz (float16 maps)
    explain_max_runs_per_capture: int = 50

    # ---------------------------------------------------------------- storage
    min_free_gb_warn: float = 2.0
    captures_page_size: int = 50

    @property
    def bind_host(self) -> str:
        return "127.0.0.1" if self.local_only else self.host

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.models_dir):
            Path(d).mkdir(parents=True, exist_ok=True)


settings = Settings()
