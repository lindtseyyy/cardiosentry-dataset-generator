"""Model registry: discover and validate model descriptors (plan §10.2).

Scans models/*.yaml, validates each against the pydantic descriptor schema,
resolves the `architecture.module` against backend/arch/, verifies the weights
file exists and hashes it, and checks `expected_sha256` — all at STARTUP, so
problems fail loudly with a readable message rather than at first inference.
"""
from __future__ import annotations

import hashlib
import importlib
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from backend.errors import ApiError
from backend.schemas.model_descriptor import ModelDescriptor
from backend.settings import settings


class RegistryError(Exception):
    pass


@dataclass
class LoadedDescriptor:
    descriptor: ModelDescriptor
    yaml_path: Path
    weights_path: Path
    checkpoint_sha256: str
    arch_module: object = field(repr=False)

    @property
    def id(self) -> str:
        return self.descriptor.id


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _active_id_from_symlink() -> str | None:
    link = settings.models_dir / "active.yaml"
    if link.is_symlink():
        return link.resolve().stem
    return None


def discover() -> dict[str, LoadedDescriptor]:
    settings.ensure_dirs()
    out: dict[str, LoadedDescriptor] = {}
    yamls = sorted(settings.models_dir.glob("*.yaml"))
    if not yamls:
        raise RegistryError(
            f"no model descriptors found in {settings.models_dir}. "
            "Write one (see models/resnet50_hires.yaml) and place the .pt in "
            "models/weights/.")
    for yp in yamls:
        if yp.name == "active.yaml":
            continue
        raw = yaml.safe_load(yp.read_text(encoding="utf-8")) or {}
        try:
            desc = ModelDescriptor.model_validate(raw)
        except Exception as exc:
            raise RegistryError(f"{yp.name}: invalid descriptor:\n  {exc}") from exc
        weights_path = (settings.models_dir / desc.weights).resolve()
        if not weights_path.is_file():
            raise RegistryError(
                f"{yp.name}: weights file not found: {weights_path}")
        if not weights_path.is_file() or weights_path.suffix != ".pt":
            raise RegistryError(f"{yp.name}: weights must be a .pt file: {weights_path}")

        # Architecture module must import and expose the registry key now.
        # Descriptors may name it as `arch.hybrids_v1` (plan-literal) or
        # `backend.arch.hybrids_v1` (fully qualified); resolve either.
        arch_module = None
        candidates = [desc.architecture.module]
        if not desc.architecture.module.startswith("backend."):
            candidates.append("backend." + desc.architecture.module)
        for mod_name in candidates:
            try:
                arch_module = importlib.import_module(mod_name)
                break
            except ModuleNotFoundError as exc:
                if exc.name != mod_name and not mod_name.startswith(exc.name + "."):
                    raise RegistryError(
                        f"{yp.name}: could not import architecture module "
                        f"{mod_name!r}:\n  {exc}") from exc
        if arch_module is None:
            raise RegistryError(
                f"{yp.name}: could not import architecture module "
                f"{desc.architecture.module!r} (tried {candidates})")
        # The REGISTRY may live on the package or on its `hybrids`/`models`
        # submodule (the vendored layout). `arch_module` must end up being the
        # module that exposes both REGISTRY and build().
        resolved_name = arch_module.__name__
        registry = getattr(arch_module, "REGISTRY", None)
        if not isinstance(registry, dict):
            for sub in ("hybrids", "models"):
                try:
                    sub_mod = importlib.import_module(f"{resolved_name}.{sub}")
                except ModuleNotFoundError:
                    continue
                reg = getattr(sub_mod, "REGISTRY", None)
                if isinstance(reg, dict):
                    registry, arch_module, resolved_name = reg, sub_mod, sub_mod.__name__
                    break
        if not isinstance(registry, dict) or desc.architecture.registry_key not in registry:
            raise RegistryError(
                f"{yp.name}: architecture module {resolved_name!r} "
                f"has no REGISTRY entry {desc.architecture.registry_key!r}")

        sha = _sha256_file(weights_path)
        if desc.expected_sha256 and desc.expected_sha256 != sha:
            raise RegistryError(
                f"{yp.name}: checkpoint sha256 mismatch — descriptor claims\n"
                f"    {desc.expected_sha256}\n  but the file is\n"
                f"    {sha}\n"
                "The .pt changed but the descriptor still claims the old hash; "
                "refusing to start (§16 case 18). Update expected_sha256 "
                "(or set it to null to accept the new file once).")

        if desc.id in out:
            raise RegistryError(f"duplicate model id {desc.id!r}")
        out[desc.id] = LoadedDescriptor(desc, yp, weights_path, sha, arch_module)
    return out


def active(loaded: dict[str, LoadedDescriptor]) -> LoadedDescriptor:
    wanted = settings.active_model or _active_id_from_symlink()
    if wanted:
        if wanted not in loaded:
            raise RegistryError(
                f"active model {wanted!r} not found among discovered descriptors "
                f"({', '.join(sorted(loaded))})")
        return loaded[wanted]
    if len(loaded) == 1:
        return next(iter(loaded.values()))
    raise RegistryError(
        "no active model: set CARDIOSENTRY_ACTIVE_MODEL or point "
        "models/active.yaml at a descriptor")
