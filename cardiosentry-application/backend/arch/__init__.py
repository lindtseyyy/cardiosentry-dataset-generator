"""Vendored model-architecture packages, pinned to the checkpoints they serve.

The `.pt` checkpoints produced by the research code are `state_dict` blobs, not
TorchScript modules: rebuilding a model requires the exact `nn.Module` classes
the training run used. These subpackages are *copies* of that code (never live
imports from prototype-code/), so the serving path cannot drift when the
research code evolves.

See README.md in this directory for provenance and the single documented
deviation from the verbatim copies.
"""
