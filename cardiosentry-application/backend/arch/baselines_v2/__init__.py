"""Round-3 plain baselines, v2 geometry study (vendored — see arch/README.md).

The nine unmodified backbones the hybrids are measured against: `build()` is a
`timm.create_model(tag, num_classes=4)` with no wrapper, so a checkpoint from
this package is a bare timm state_dict (`conv_stem…classifier`) rather than the
`backbone.*`/`attn.*` blob `hybrids_v2` produces. REGISTRY and `build()` live in
`models`; `config`/`dimensions` carry the constants it reads and are
byte-identical to the `hybrids_v2` copies (the two research packages share
them), kept here so this package is pinned on its own.
"""
from . import config, dimensions, models  # noqa: F401
