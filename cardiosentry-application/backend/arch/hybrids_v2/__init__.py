"""Round-3 hybrid architectures, v2 geometry study (vendored — see arch/README.md).

The REGISTRY and the `build()` factory live in `models`; `hybrids` holds the
`nn.Module` classes; `config`/`dimensions` carry the constants those two read.
"""
from . import config, dimensions, hybrids, models  # noqa: F401
