#!/usr/bin/env python3
"""
Decide whether a micro-batch override is legitimate for the models about to run.

    python check_micro_batch.py 8 convnext_base_cbam vit_b16_cbam ...

Exits 0 and prints what the override means, or exits 1 and says why not.
`run_all.sh --micro-batch` calls this before it will pass the flag through.

WHY THIS IS A GATE AND NOT A COMMENT

micro_batch is a VRAM accommodation, not an experimental variable - gradient
accumulation restores config.EFFECTIVE_BATCH = 32, and steps_per_epoch (and so
the entire cosine schedule and the 2-epoch warmup) is 151 at any of 2, 4, 8
or 16. On a
network without normalisation over the batch, changing it is genuinely free: the
accumulated gradient is the same sum, to floating-point associativity.

BatchNorm breaks that, and breaks it invisibly. BN normalises over the MICRO-
batch; accumulation sums gradients, it does not merge batch statistics. Change
micro_batch on a BN network and you have changed what the network computes.

In this package that is worse than a fairness question, because every hybrid is
reported as a DELTA against the baselines_v2 run of its backbone, trained at the
registry value. A hybrid trained at a different micro_batch is measuring the
normalisation as well as the component, the loss curve looks entirely normal, and
no metric in the report would reveal it. So the flag is refused rather than
warned about.

    densenet121      121 BatchNorm layers   -> refused
    efficientnet_b3   78 BatchNorm layers   -> refused
    convnext_base      0 (LayerNorm)        -> free
    vit_b16            0 (LayerNorm)        -> free
    swinv2_base        0 (LayerNorm)        -> free

The counts come from models.BACKBONES and selftest.py --build asserts them
against the real networks, so a stale number cannot licence the confound this
exists to prevent.
"""
from __future__ import annotations
import sys

import config as C
import models


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 1
    try:
        mb = int(argv[0])
    except ValueError:
        print(f"--micro-batch takes a number, got '{argv[0]}'")
        return 1
    names = argv[1:]

    unknown = [n for n in names if n not in models.REGISTRY]
    if unknown:
        print(f"unknown model(s): {unknown}")
        return 1

    if mb < 1 or C.EFFECTIVE_BATCH % mb:
        print(f"REFUSING --micro-batch {mb}: it must divide EFFECTIVE_BATCH "
              f"({C.EFFECTIVE_BATCH}) exactly, or the effective batch differs "
              f"between runs and the comparison stops being controlled.")
        ok = [d for d in range(1, C.EFFECTIVE_BATCH + 1) if C.EFFECTIVE_BATCH % d == 0]
        print(f"  valid values: {ok}")
        return 1

    # Only a CHANGE is a problem. Passing the value the registry already uses is
    # a no-op, and refusing it would mean `run_all.sh --micro-batch 8` failed on
    # efficientnet_b3 for asking for exactly what it was going to do.
    bn = {}
    for n in names:
        b = models.REGISTRY[n]["backbone"]
        if models.BACKBONES[b]["batchnorm"] and models.BACKBONES[b]["micro_batch"] != mb:
            bn[b] = models.BACKBONES[b]["batchnorm"]
    if bn:
        for b, n_bn in sorted(bn.items()):
            print(f"REFUSING --micro-batch {mb}: {b} has {n_bn} BatchNorm layers "
                  f"(registry value {models.BACKBONES[b]['micro_batch']}).")
        print("  BatchNorm normalises over the MICRO-batch, and gradient")
        print("  accumulation sums gradients - it does not merge batch statistics.")
        print("  Changing it changes what the network COMPUTES, and every run here")
        print("  is scored as a delta against a baselines_v2 baseline trained at")
        print("  the registry value. The loss curve would look completely normal.")
        print("  Leave it, or re-train that backbone's baseline at the new value.")
        return 1

    fams = sorted({models.REGISTRY[n]["backbone"] for n in names})
    same = [f for f in fams if models.BACKBONES[f]["micro_batch"] == mb]
    print(f"micro-batch override accepted: {mb} "
          f"(accum {C.EFFECTIVE_BATCH // mb}, effective batch "
          f"{C.EFFECTIVE_BATCH} unchanged)")
    if len(same) == len(fams):
        print(f"  this is already the registry value for "
              f"{', '.join(fams)} - no change.")
    else:
        changed = [f for f in fams if f not in same]
        for f in changed:
            print(f"  {f}: {models.BACKBONES[f]['micro_batch']} -> {mb}, and it has "
                  f"no BatchNorm, so the")
            print(f"  accumulated gradient is identical and steps_per_epoch does "
                  f"not move.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
