#!/usr/bin/env python3
"""
Train ONE round-2 architecture under the protocol in config.py.

    python train.py --model resnet50_dual --data /content/prototype_1152_768

Run once per architecture, then `python compare.py --runs <dir>` to tabulate.
compare.py verifies the PROTOCOL fingerprints match before it will compare them;
input resolution is expected to differ and is reported rather than enforced.

Colab-safe: checkpoints every epoch and resumes automatically after a disconnect.
"""
from __future__ import annotations
import argparse, json, os, platform, time
from pathlib import Path

import numpy as np
import torch

import artifacts
import af_lvh
import baseline
import config as C
import data as D
import engine
import hybrids
import manifest
import metrics


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=hybrids.MODEL_NAMES)
    p.add_argument("--data", required=True,
                   help=f"prototype folder pre-resized to {C.LOAD_SIZE}px "
                        f"(see prepare_dataset.py)")
    p.add_argument("--out", default="runs", help="where to write runs/<model>/")
    p.add_argument("--epochs", type=int, default=None, help="override config.EPOCHS")
    p.add_argument("--lr", type=float, default=None,
                   help="override config.LR - BREAKS the controlled comparison "
                        "unless you do the same for every model")
    p.add_argument("--micro-batch", type=int, default=None,
                   help="VRAM accommodation only; effective batch stays constant")
    p.add_argument("--workers", type=int, default=0,
                   help="0 = auto: min(cpu_count, 4). Decoding is CPU-bound, so "
                        "more workers than cores buys nothing and torch warns "
                        "about it. Pass a number to override on a high-CPU VM.")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--no-channels-last", action="store_true",
                   help="disable NHWC. Layout only - use it to check the "
                        "optimisation is not perturbing anything, not routinely.")
    p.add_argument("--smoke", action="store_true",
                   help="20 train / 10 eval batches, 1 epoch - proves the whole "
                        "path works in ~2 min before you commit hours")
    p.add_argument("--sweep", action="store_true",
                   help="equal-budget LR search over config.LR_GRID; every "
                        "architecture gets the same candidates and the same rule")
    p.add_argument("--no-ablation", action="store_true",
                   help="skip the per-stream test re-scoring. Do not use: it is "
                        "the measurement this round exists for, and it costs one "
                        "extra pass over 938 images")
    p.add_argument("--keep-last", action="store_true",
                   help="keep last.pt after a model finishes. It is only needed "
                        "to RESUME an unfinished run and is ~16 bytes/param, so "
                        "it is deleted on success by default to protect Drive.")
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    return p.parse_args()


def device_info():
    if torch.cuda.is_available():
        pr = torch.cuda.get_device_properties(0)
        return {"device": pr.name, "vram_gb": round(pr.total_memory / 1024**3, 1),
                "cuda": torch.version.cuda}
    return {"device": "cpu", "vram_gb": 0, "cuda": None}


def run_training(args, base_lr, loaders, device, tag_suffix="", quiet=False, art=None):
    """One full training run at one LR. Returns (model, best_state, history, best_val, info)."""
    engine.set_seed(C.SEED)

    model, minfo = hybrids.build(args.model, pretrained=not args.no_pretrained)
    model.to(device)

    # NHWC for the convolutional models. Layout only: same convolutions, same
    # numbers, arranged the way Ada/Ampere tensor cores read them. Ignored on
    # CPU and worth nothing pre-Turing, so it is safe to leave on everywhere.
    use_cl = bool(minfo.get("channels_last")) and not args.no_channels_last \
             and device.type == "cuda"
    if use_cl:
        model = model.to(memory_format=torch.channels_last)
    model.channels_last = use_cl
    minfo["channels_last"] = use_cl

    accum = hybrids.accumulation_steps(args.model, args.micro_batch)
    criterion = engine.build_criterion(loaders["pos_weight"], device)
    optimizer = engine.build_optimizer(model, base_lr)
    scaler = torch.amp.GradScaler("cuda", enabled=C.AMP and device.type == "cuda")

    epochs = 1 if args.smoke else (args.epochs or C.EPOCHS)
    max_tb = 20 if args.smoke else None
    max_eb = 10 if args.smoke else None

    ckpt_dir = Path(args.out) / args.model / "weights"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_ckpt = ckpt_dir / f"last{tag_suffix}.pt"

    start_epoch, history, best_val, best_state, bad = 0, [], -np.inf, None, 0

    if args.resume and last_ckpt.exists() and not args.smoke:
        blob = torch.load(last_ckpt, map_location=device, weights_only=False)
        data_root = str(Path(args.data).resolve())
        if blob.get("protocol_fingerprint") != C.protocol_fingerprint():
            print(f"  checkpoint protocol {blob.get('protocol_fingerprint')} != "
                  f"current {C.protocol_fingerprint()} - starting fresh")
        elif blob.get("views") not in (None, list(model.views)):
            # Half a run at one resolution and half at another is silently invalid.
            print(f"  checkpoint used views {blob.get('views')} but this model "
                  f"wants {list(model.views)} - starting fresh")
        elif blob.get("data_root") not in (None, data_root):
            print(f"  checkpoint was trained on {blob.get('data_root')}\n"
                  f"  but --data is {data_root} - starting fresh")
        else:
            model.load_state_dict(blob["model"])
            optimizer.load_state_dict(blob["optimizer"])
            scaler.load_state_dict(blob["scaler"])
            start_epoch = blob["epoch"] + 1
            history = blob["history"]
            best_val = blob["best_val"]
            best_state = blob.get("best_state")
            bad = blob.get("bad", 0)
            print(f"  resumed from epoch {start_epoch}/{epochs}")

    FOC = C.FOCUS_LABEL

    for ep in range(start_epoch, epochs):
        if not quiet:
            print(f"\n  epoch {ep+1}/{epochs}  (lr base {base_lr:.1e})", flush=True)
        tr = engine.train_one_epoch(model, loaders["train"], optimizer, criterion,
                                    scaler, device, ep, epochs, accum, base_lr,
                                    max_batches=max_tb,
                                    log_every=0 if quiet else 50)
        val, thr, _, _ = engine.validate(model, loaders["val"], device,
                                         max_batches=max_eb)
        score = val[C.MONITOR]
        row = {"epoch": ep + 1, "train_loss": tr["loss"],
               "seconds": round(tr["seconds"], 1), "lr": base_lr,
               "img_per_sec": round(tr.get("img_per_sec", 0), 1),
               "data_wait_pct": round(tr.get("data_wait_pct", 0), 1),
               "val_macro_auroc": val["macro_auroc"],
               "val_macro_auprc": val["macro_auprc"],
               "val_macro_f1": val["macro_f1"],
               "val_macro_sensitivity": val["macro_sensitivity"],
               "val_macro_specificity": val["macro_specificity"],
               "val_macro_precision": val["macro_precision"],
               "val_exact_match_acc": val["exact_match_accuracy"],
               "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
        # EVERY head, every epoch - not just the focus one. A model that lifts
        # macro AUPRC while quietly degrading STEMI looks fine in the macro
        # column and is not fine; this is where that becomes visible while the
        # run is still going, rather than in the post-mortem.
        for _l in C.LABELS:
            row[f"val_{_l}_auprc"] = val["per_label"][_l]["auprc"]
            row[f"val_{_l}_sens"] = val["per_label"][_l]["sensitivity"]
            row[f"val_{_l}_fpr"] = val["per_label"][_l]["fpr"]
        history.append(row)
        if art is not None:
            art.append_epoch(row)          # flushed now, survives a crash
            art.set_status("RUNNING", f"epoch {ep+1}/{epochs}  "
                                      f"best val {C.MONITOR} {max(best_val, score):.4f}")
        if not quiet:
            print(f"    train loss {tr['loss']:.4f}  |  val mAUPRC "
                  f"{val['macro_auprc']:.4f}  |  "
                  + "  ".join(f"{l} {val['per_label'][l]['auprc']:.3f}"
                              for l in C.LABELS)
                  + f"  ({tr['seconds']/60:.1f} min)", flush=True)

        improved = score > best_val
        if improved:
            best_val, bad = score, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save({"model": best_state, "thresholds": thr, "epoch": ep,
                        "val": val, "views": list(model.views),
                        "protocol_fingerprint": C.protocol_fingerprint()},
                       ckpt_dir / f"best{tag_suffix}.pt")
        else:
            bad += 1

        if not args.smoke:
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict(), "epoch": ep, "history": history,
                        "best_val": best_val, "best_state": best_state, "bad": bad,
                        "views": list(model.views),
                        "protocol_fingerprint": C.protocol_fingerprint(),
                        "data_root": str(Path(args.data).resolve())}, last_ckpt)

        if bad >= C.EARLY_STOP_PATIENCE and not args.smoke:
            print(f"    early stop: no {C.MONITOR} improvement in "
                  f"{C.EARLY_STOP_PATIENCE} epochs")
            break

    return model, best_state, history, best_val, minfo


def build_report(result, val_res, test_res, per_src, comp, bench, history,
                 ablations, extra, af_row=None) -> str:
    """The file you read in the morning. Everything, in plain text."""
    FOC = C.FOCUS_LABEL
    L = []
    A = L.append
    A("=" * 92)
    A(f"CardioSentry hybrids  |  {result['model']}")
    A("=" * 92)
    A(f"purpose           {result['model_info']['why']}")
    A(f"finished          {result['finished_utc']}")
    A(f"training time     {result['train_seconds']/60:.1f} min over "
      f"{result['epochs_run']} epoch(s)")
    A(f"protocol fp       {result['protocol_fingerprint']}   <- must match across "
      f"all compared runs")
    A(f"input views       {result['model_info']['views']}   (fp "
      f"{result['model_info']['input_fingerprint']})   <- EXPECTED to differ; "
      f"it is the variable")
    A(f"feature groups    {result['model_info']['groups']}")
    A(f"dataset           {result['data']['root']}")
    A(f"                  {result['data']['rows']:,} images | "
      f"{result['data']['records']:,} records | {result['data']['patients']:,} patients")
    A(f"backbone tags     {result['model_info']['tags']}")
    A(f"pretraining       {result['model_info']['pretrain']}")
    A(f"learning rate     {result['lr_used']:.1e}"
      + ("  (equal-budget sweep)" if result['sweep'] else ""))
    A("")
    A("-" * 92)
    A("MEASUREMENT PRECISION AT THIS INPUT SIZE")
    A("-" * 92)
    for v in result["model_info"]["views"]:
        A(f"  {v:>4}px view:  {extra['px_per_mm'][str(v)]['vertical']:.2f} px/mm "
          f"vertical ({extra['px_per_mm'][str(v)]['mv_px']:.1f} px per mV)   "
          f"{extra['px_per_mm'][str(v)]['horizontal']:.2f} px/mm horizontal")
    A("  Source sheets render at 150 DPI = 5.91 px/mm, 10 mm/mV, 25 mm/s.")
    A(f"  Sokolow-Lyon and Cornell turn on differences of a few mm, so these")
    A(f"  numbers are the hypothesis under test.")
    A("")
    A("-" * 92)
    A("MODEL COMPLEXITY")
    A("-" * 92)
    A(f"  parameters (total)      {comp['params_total']:>15,}   ({comp['params_total_m']:.2f} M)")
    A(f"  parameters (trainable)  {comp['params_trainable']:>15,}   ({comp['params_trainable_m']:.2f} M)")
    A(f"  model size              {comp['size_mb']:>15.2f} MB  (fp32 params + buffers)")
    A("")
    A("-" * 92)
    A("INFERENCE SPEED")
    A("-" * 92)
    A(f"  measured on {bench['device']}  |  views {bench['views']}  |  "
      f"AMP {'on' if bench['amp'] else 'off'}  |  {bench['reps']} reps")
    A(f"  single image (batch=1)  {bench['latency_ms_b1_mean']:8.2f} ms  "
      f"+/- {bench['latency_ms_b1_std']:.2f}   p50 {bench['latency_ms_b1_p50']:.2f}  "
      f"p95 {bench['latency_ms_b1_p95']:.2f}")
    A(f"                          {bench['throughput_ips_b1']:8.1f} images/sec")
    A(f"  batched (batch={bench['batch_size']})        "
      f"{bench['ms_per_image_batched']:8.2f} ms/image")
    A(f"                          {bench['throughput_ips_batched']:8.1f} images/sec")
    A("  Synthetic input: excludes disk and JPEG decode. That exclusion is larger")
    A(f"  this round - decoding a {C.LOAD_SIZE}px sheet costs 4x a 384px one - so")
    A("  quote img/s from the training log for an end-to-end figure.")
    A("")
    A("-" * 92)
    A("VALIDATION  (thresholds tuned here, then frozen)")
    A("-" * 92)
    A(metrics.format_report(val_res))
    A("")
    A("-" * 92)
    A("TEST  (val-tuned thresholds, frozen before test - no tuning on test)")
    A("-" * 92)
    A(metrics.format_report(test_res))
    A("")
    A("-" * 92)
    A(f"{C.FOCUS_LABEL} FOCUS  -  the head this round exists to improve")
    A("-" * 92)
    f = test_res["per_label"][FOC]
    A(f"  AUPRC        {f['auprc']:.4f}          AUROC       {f['auroc']:.4f}")
    A(f"  sensitivity  {f['sensitivity']:.4f}          specificity {f['specificity']:.4f}")
    A(f"  precision    {f['precision']:.4f}          F1          {f['f1']:.4f}")
    A(f"  FPR          {f['fpr']:.4f}          threshold   {f['threshold']:.2f}")
    A(f"  TP {f['tp']:>4}   FP {f['fp']:>4}   FN {f['fn']:>4}   TN {f['tn']:>4}"
      f"     ({f['support']} positives of {f['n']})")
    A("")
    A(f"  Round 1 reference, best of five backbones on {C.FOCUS_LABEL}:")
    A("      AUPRC 0.8822 (ViT-B/384)    sensitivity 0.8784 (VGG-16)")
    A("      FPR   0.0747 (ConvNeXt-B)   ResNet-50: AUPRC 0.8805, sens 0.8078, FPR 0.0791")
    A(f"  Round 1's {C.FOCUS_LABEL} failure was FALSE-POSITIVE dominated (333 FP vs")
    A("  229 FN over five models). A model that lifts sensitivity while lifting")
    A("  FPR has moved the threshold, not solved the problem.")
    A("")
    A("-" * 92)
    A("PER-CONDITION REGRESSION CHECK  -  did this cost any other head?")
    A("-" * 92)
    A("  Macro averages dilute: a 0.04 loss on one head moves the macro by 0.01")
    A("  and reads as noise. This round targets LVH, so the other three heads")
    A("  need checking explicitly - especially STEMI, where a miss is not")
    A("  recoverable by a follow-up echo the way an over-called LVH is.")
    A("")
    A(f"  vs round 1's {baseline.REFERENCE} on the same frozen test split")
    A(f"  {'head':<9}{'AUPRC':>9}{'d':>9}{'Sens':>9}{'d':>9}"
      f"{'FPR':>9}{'d':>9}{'FN':>6}{'r1 FN':>7}")
    flags = []
    for lab in C.LABELS:
        d = test_res["per_label"][lab]
        r_ap = baseline.reference("auprc", lab)
        r_se = baseline.reference("sensitivity", lab)
        r_fp = baseline.reference("fpr", lab)
        A(f"  {lab:<9}{d['auprc']:9.4f}{d['auprc']-r_ap:+9.4f}"
          f"{d['sensitivity']:9.4f}{d['sensitivity']-r_se:+9.4f}"
          f"{d['fpr']:9.4f}{d['fpr']-r_fp:+9.4f}"
          f"{d['fn']:6d}{baseline.COUNTS[lab]['fn']:7d}")
        if d["auprc"] - r_ap < -0.02:
            flags.append(f"{lab} AUPRC down {r_ap - d['auprc']:.4f}")
        if lab == "STEMI" and d["sensitivity"] - r_se < -0.01:
            n = d["fn"]
            flags.append(f"STEMI sensitivity down {r_se - d['sensitivity']:.4f} "
                         f"- {n} missed infarct{'' if n == 1 else 's'} against "
                         f"round 1's {baseline.COUNTS['STEMI']['fn']}")
    A("")
    if flags:
        A("  FLAGS:")
        for f_ in flags:
            A(f"    - {f_}")
        A("  A model that improves LVH and trips one of these is not a better model.")
    else:
        A("  no head lost more than 0.02 AUPRC, and STEMI sensitivity held within")
        A("  0.01 of round 1.")
    A("")
    if ablations:
        A("-" * 92)
        A("STREAM ABLATION  -  what each feature group was actually worth")
        A("-" * 92)
        A(engine.format_stream_ablation(test_res, ablations))
        A("")
    if per_src:
        A("-" * 92)
        A("SOURCE CONFOUND CHECK")
        A("-" * 92)
        A(metrics.format_per_source(per_src))
        vals = [d["macro_auroc"] for d in per_src.values()]
        spread = max(vals) - min(vals) if len(vals) > 1 else 0.0
        A("")
        A("  All STEMI records come from a single database, and stage 5 of the image")
        A("  pipeline measured residual source signal (probe 0.515 vs 0.372 chance).")
        A(f"  Macro AUROC spread across sources: {spread:.4f}"
          + ("   <- LARGE, investigate" if spread > 0.10 else "   (moderate)"))
        A("  Watch this especially for the FPN and high-resolution runs: both give")
        A("  the classifier more access to fine texture, and on a rendered sheet")
        A("  fine texture includes the grid colour and JPEG signature that identify")
        A("  the source database. A resolution gain that arrives together with a")
        A("  widening source spread is a leakage gain, not a diagnostic one.")
        A("")
    if af_row is not None:
        A("-" * 92)
        A("AF x LVH SUBGROUP  -  does LVH survive when AF is present?")
        A("-" * 92)
        A("  subgroups by GROUND TRUTH on the frozen test split; decisions are this")
        A("  run's OWN frozen thresholds; AUPRC is the reference-pool design.")
        A("  THE FINDING THIS RUN IS SCORED AGAINST (af_lvh_joint_v1): the 768")
        A("  reference detects AF but missed 10 of the 28 AF+LVH cases' LVH.")
        A("")
        for line in af_lvh.format_run_block(
                af_row, "+".join(map(str, result["model_info"]["views"]))):
            A(line)
        A("")
    A("-" * 92)
    A("PER-EPOCH HISTORY")
    A("-" * 92)
    A("  per-head val AUPRC every epoch, so a head that degrades while the macro")
    A("  improves is visible here rather than only in the post-mortem.")
    A(f"  {'ep':>3}{'train_loss':>12}{'mAUPRC':>10}"
      + "".join(f"{l:>10}" for l in C.LABELS) + f"{'min':>8}")
    for h in history:
        A(f"  {h['epoch']:>3}{h['train_loss']:12.4f}{h['val_macro_auprc']:10.4f}"
          + "".join(f"{h.get('val_'+l+'_auprc', float('nan')):10.4f}"
                    for l in C.LABELS)
          + f"{h['seconds']/60:8.1f}")
    A("")
    if extra.get("cbam_gammas"):
        A("-" * 92)
        A("LEARNED CBAM GATE STRENGTHS")
        A("-" * 92)
        A("  Each CBAM starts at gamma=0, i.e. exactly the identity, so these are")
        A("  what the network CHOSE. Values near zero mean attention was offered")
        A("  and declined - a real result, and evidence that reweighting existing")
        A("  features was not the bottleneck.")
        for n, v in extra["cbam_gammas"].items():
            A(f"    {n:<52}{v:+.4f}")
        A("")
    A("-" * 92)
    A("NOTES")
    A("-" * 92)
    A("  * Prototype subset: train is subsampled at 1 augmentation variant per record.")
    A("    AUROC/AUPRC and the RANKING of architectures transfer to the full corpus;")
    A("    thresholds, calibration and absolute scores do not.")
    A("  * The head is a 2-layer MLP for EVERY model in this round, including the")
    A("    baselines. Round 1 used a single Linear, so round-1 numbers are a")
    A("    reference point, not a control. resnet50_base is the control.")
    A("  * 'accuracy' for a multi-label model is ambiguous. Both readings are given:")
    A("    exact-match (all 4 heads right on one ECG) and hamming (per label slot).")
    A("=" * 92)
    return "\n".join(L)


def px_per_mm(views) -> dict:
    """Measurement precision at each view size. 150 DPI source, 1686x1311 sheet."""
    src_dpi_px_per_mm = 150 / 25.4          # 5.906
    out = {}
    for v in views:
        vert = src_dpi_px_per_mm * (v / 1311)
        horiz = src_dpi_px_per_mm * (v / 1686)
        out[str(v)] = {"vertical": vert, "horizontal": horiz, "mv_px": vert * 10}
    return out


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    art = artifacts.RunArtifacts(args.out, args.model, resume=args.resume)
    try:
        print(f"protocol fingerprint: {C.protocol_fingerprint()}   "
              f"(must match across all runs)")
        views = hybrids.views(args.model)
        print(f"input views: {list(views)}   (this is the experimental variable)")
        dinfo = device_info()
        print(f"device: {dinfo['device']} ({dinfo['vram_gb']} GB)")
        if device.type == "cpu":
            print("WARNING: no GPU. In Colab: Runtime > Change runtime type > GPU")

        mb = hybrids.micro_batch(args.model, args.micro_batch)
        accum = hybrids.accumulation_steps(args.model, args.micro_batch)
        print(f"batching: micro {mb} x accum {accum} = effective "
              f"{C.EFFECTIVE_BATCH} (identical for every model)")

        print("\nloading dataset ...")
        t0 = time.time()
        n_workers = D.resolve_workers(args.workers)
        print(f"dataloader: {n_workers} worker(s) on {os.cpu_count()} CPU(s)")
        loaders = D.build_loaders(args.data, mb, views, n_workers)
        print(manifest.describe(loaders["df"]))
        print(f"  integrity checks passed in {time.time()-t0:.1f}s")

        # ---- train ---------------------------------------------------------
        started = time.time()
        if args.sweep:
            print(f"\nequal-budget LR sweep over {C.LR_GRID}")
            results = []
            for lr in C.LR_GRID:
                print(f"\n--- lr {lr:.0e} ---")
                m, st, hist, bv, minfo = run_training(args, lr, loaders, device,
                                                      tag_suffix=f"_lr{lr:.0e}",
                                                      quiet=True, art=art)
                print(f"    best val {C.MONITOR}: {bv:.4f}")
                results.append((bv, lr, m, st, hist, minfo))
            best_val, base_lr, model, best_state, history, minfo = max(
                results, key=lambda r: r[0])
            print(f"\nselected lr {base_lr:.0e} (val {C.MONITOR} {best_val:.4f})")
        else:
            base_lr = args.lr or C.LR
            model, best_state, history, best_val, minfo = run_training(
                args, base_lr, loaders, device, art=art)
        elapsed = time.time() - started

        # ---- final evaluation ----------------------------------------------
        if best_state is not None:
            model.load_state_dict(best_state)
        max_eb = 10 if args.smoke else None

        print("\n" + "=" * 78)
        val_res, thresholds, vp, vy = engine.validate(model, loaders["val"], device,
                                                      max_batches=max_eb)
        print(metrics.format_report(val_res, "VALIDATION (thresholds tuned here)"))

        tp, ty, tidx = engine.predict(model, loaders["test"], device, max_batches=max_eb)
        test_res = metrics.evaluate(ty, tp, thresholds)
        print()
        print(metrics.format_report(test_res, "TEST (val-tuned thresholds, frozen)"))

        ablations = {}
        if C.STREAM_ABLATION and not args.no_ablation:
            ablations = engine.evaluate_streams(model, loaders["test"], device,
                                                thresholds, max_batches=max_eb)
            if ablations:
                print("\nSTREAM ABLATION")
                print(engine.format_stream_ablation(test_res, ablations))

        per_src = {}
        if C.REPORT_PER_SOURCE:
            src = loaders["test_df"].iloc[tidx]["source"].tolist()
            per_src = metrics.evaluate_per_source(ty, tp, src, thresholds)
            print()
            print(metrics.format_per_source(per_src))
            a = [d["macro_auroc"] for d in per_src.values()]
            if len(a) > 1 and (max(a) - min(a)) > 0.10:
                print(f"\n  WARNING: macro AUROC varies by {max(a)-min(a):.3f} across "
                      f"sources - the model may be keying on acquisition artifacts.")

        # ---- complexity + speed --------------------------------------------
        print("\nbenchmarking inference ...")
        comp = engine.model_complexity(model)
        bench = engine.benchmark_inference(model, device, views, batch_size=mb,
                                           reps=10 if args.smoke else 30)
        print(f"  {comp['params_total_m']:.1f} M params | {comp['size_mb']:.1f} MB | "
              f"{bench['latency_ms_b1_mean']:.2f} ms/image (batch=1) | "
              f"{bench['throughput_ips_batched']:.1f} img/s batched")

        extra = {"px_per_mm": px_per_mm(views)}
        if hasattr(model, "gammas"):
            extra["cbam_gammas"] = model.gammas()

        # ---- persist --------------------------------------------------------
        result = {
            "model": args.model,
            "protocol_fingerprint": C.protocol_fingerprint(),
            "input_fingerprint": minfo["input_fingerprint"],
            "protocol_settings": C.protocol_settings(),
            "input_settings": C.input_settings(),
            "model_info": minfo,
            "complexity": comp,
            "inference": bench,
            "data": {"root": str(Path(args.data).resolve()), **loaders["report"]},
            "pos_weight": loaders["pos_weight"],
            "micro_batch": mb,
            "accum_steps": accum,
            "lr_used": base_lr,
            "sweep": bool(args.sweep),
            "smoke": bool(args.smoke),
            "epochs_run": len(history),
            "best_val_monitor": float(best_val),
            "history": history,
            "val": val_res,
            "test": test_res,
            "test_stream_ablation": ablations,
            "test_per_source": per_src,
            "thresholds": thresholds,
            "extra": extra,
            "train_seconds": elapsed,
            "env": {**dinfo, "torch": torch.__version__,
                    "python": platform.python_version()},
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        (art.dir / "result.json").write_text(json.dumps(result, indent=2, default=str))

        # Predictions first: the AF x LVH subgroup block in report.txt is scored
        # from the same per-image file compare.py will later read, so the two
        # can never disagree.
        if not args.smoke:
            art.save_predictions("val", loaders["val_df"], vy, vp, thresholds)
            art.save_predictions("test",
                                 loaders["test_df"].iloc[tidx].reset_index(drop=True),
                                 ty, tp, thresholds)

        af_row = af_lvh.score_run(art.dir) if not args.smoke else None
        art.write_text("report.txt", build_report(result, val_res, test_res, per_src,
                                                  comp, bench, history, ablations, extra,
                                                  af_row))
        art.write_csv("metrics.csv",
                      metrics.flat_rows(val_res, "val", args.model)
                      + metrics.flat_rows(test_res, "test", args.model))

        print("\n" + (art.dir / "report.txt").read_text())

        # last.pt exists only to resume an interrupted run. This run finished, so
        # it is now dead weight in Drive. best.pt has the weights that matter.
        if not args.keep_last and not args.smoke:
            freed = 0
            for f in (art.dir / "weights").glob("last*.pt"):
                freed += f.stat().st_size
                f.unlink()
            if freed:
                print(f"removed resume checkpoint(s), freed {freed/1024**3:.2f} GB "
                      f"(pass --keep-last to retain them)")

        art.finalize(ok=True, note=f"best val {C.MONITOR} = {best_val:.4f}")
        artifacts.write_summary(args.out)
        if args.smoke:
            print("\nSMOKE TEST ONLY - metrics are meaningless. Re-run without --smoke.")
        return 0

    except BaseException as exc:          # includes KeyboardInterrupt
        art.fail(exc)
        try:
            artifacts.write_summary(args.out)
        except Exception:
            pass
        art.finalize(ok=False, note=f"{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
