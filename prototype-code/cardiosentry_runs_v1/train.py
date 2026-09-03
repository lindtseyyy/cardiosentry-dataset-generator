#!/usr/bin/env python3
"""
Train ONE architecture under the controlled conditions in config.py.

    python train.py --model resnet50 --data /content/prototype_1152

Run it once per architecture, then `python compare.py --runs runs/` to tabulate.
compare.py verifies the config fingerprints match before it will compare them.

Colab-safe: checkpoints every epoch and resumes automatically after a disconnect.
"""
from __future__ import annotations
import argparse, json, os, platform, sys, time
from pathlib import Path

import numpy as np
import torch

import artifacts
import config as C
import data as D
import engine
import manifest
import metrics
import models


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=models.MODEL_NAMES)
    p.add_argument("--data", required=True, help="extracted prototype folder")
    p.add_argument("--out", default="runs", help="where to write runs/<model>/")
    p.add_argument("--epochs", type=int, default=None, help="override config.EPOCHS")
    p.add_argument("--lr", type=float, default=None,
                   help="override config.LR - BREAKS the controlled comparison "
                        "unless you do the same for every model")
    p.add_argument("--micro-batch", type=int, default=None,
                   help="VRAM accommodation only; effective batch stays constant")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--smoke", action="store_true",
                   help="20 train / 10 eval batches, 1 epoch - proves the whole "
                        "path works in ~2 min before you commit hours")
    p.add_argument("--sweep", action="store_true",
                   help="equal-budget LR search over config.LR_GRID; every "
                        "architecture gets the same candidates and the same rule")
    p.add_argument("--keep-last", action="store_true",
                   help="keep last.pt after a model finishes. It is only needed "
                        "to RESUME an unfinished run and is ~16 bytes/param "
                        "(2 GB for VGG-16), so it is deleted on success by "
                        "default to protect your Drive quota.")
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
    """One full training run at one LR. Returns (best_state, history, best_val)."""
    engine.set_seed(C.SEED)

    model, minfo = models.build(args.model, pretrained=not args.no_pretrained)
    model.to(device)

    accum = models.accumulation_steps(args.model, args.micro_batch)
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
        if blob.get("fingerprint") != C.fingerprint():
            print(f"  checkpoint fingerprint {blob.get('fingerprint')} != current "
                  f"{C.fingerprint()} - config changed, starting fresh")
        elif blob.get("data_root") not in (None, data_root):
            # Resuming across a dataset change would give half a run on one
            # corpus and half on another - silently invalid.
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
        history.append(row)
        if art is not None:
            art.append_epoch(row)          # flushed now, survives a crash
            art.set_status("RUNNING", f"epoch {ep+1}/{epochs}  "
                                      f"best val {C.MONITOR} {max(best_val, score):.4f}")
        if not quiet:
            print(f"    train loss {tr['loss']:.4f}  |  val AUROC "
                  f"{val['macro_auroc']:.4f}  AUPRC {val['macro_auprc']:.4f}  "
                  f"({tr['seconds']/60:.1f} min)", flush=True)

        improved = score > best_val
        if improved:
            best_val, bad = score, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save({"model": best_state, "thresholds": thr, "epoch": ep,
                        "val": val, "fingerprint": C.fingerprint()},
                       ckpt_dir / f"best{tag_suffix}.pt")
        else:
            bad += 1

        if not args.smoke:
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict(), "epoch": ep, "history": history,
                        "best_val": best_val, "best_state": best_state, "bad": bad,
                        "fingerprint": C.fingerprint(),
                        "data_root": str(Path(args.data).resolve())}, last_ckpt)

        if bad >= C.EARLY_STOP_PATIENCE and not args.smoke:
            print(f"    early stop: no {C.MONITOR} improvement in "
                  f"{C.EARLY_STOP_PATIENCE} epochs")
            break

    return model, best_state, history, best_val, minfo


def build_report(args, result, val_res, test_res, per_src, comp, bench,
                 loaders, history) -> str:
    """The file you read in the morning. Everything, in plain text."""
    L = []
    A = L.append
    A("=" * 92)
    A(f"CardioSentry  |  {result['model']}")
    A("=" * 92)
    A(f"finished          {result['finished_utc']}")
    A(f"training time     {result['train_seconds']/60:.1f} min over {result['epochs_run']} epoch(s)")
    A(f"config fingerprint {result['fingerprint']}   <- must match across all compared runs")
    A(f"dataset           {result['data']['root']}")
    A(f"                  {result['data']['rows']:,} images | "
      f"{result['data']['records']:,} records | {result['data']['patients']:,} patients")
    A(f"pretrained tag    {result['model_info']['resolved_tag']}")
    A(f"pretraining       {result['model_info']['pretrain']}")
    A(f"learning rate     {result['lr_used']:.1e}" + ("  (equal-budget sweep)" if result['sweep'] else ""))
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
    A(f"  measured on {bench['device']}  |  {bench['img_size']}x{bench['img_size']}  |  "
      f"AMP {'on' if bench['amp'] else 'off'}  |  {bench['reps']} reps")
    A(f"  single image (batch=1)  {bench['latency_ms_b1_mean']:8.2f} ms  "
      f"+/- {bench['latency_ms_b1_std']:.2f}   p50 {bench['latency_ms_b1_p50']:.2f}  "
      f"p95 {bench['latency_ms_b1_p95']:.2f}")
    A(f"                          {bench['throughput_ips_b1']:8.1f} images/sec")
    A(f"  batched (batch={bench['batch_size']})        "
      f"{bench['ms_per_image_batched']:8.2f} ms/image")
    A(f"                          {bench['throughput_ips_batched']:8.1f} images/sec")
    A("  synthetic input: excludes disk and JPEG decode, so this reflects the")
    A("  architecture rather than the storage medium.")
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
        A("  Report these per-source numbers next to the headline, whatever they show.")
        A("")
    A("-" * 92)
    A("PER-EPOCH HISTORY")
    A("-" * 92)
    A(f"  {'ep':>3}{'train_loss':>12}{'val_mAUROC':>12}{'val_mAUPRC':>12}"
      f"{'val_mF1':>10}{'min':>8}")
    for h in history:
        A(f"  {h['epoch']:>3}{h['train_loss']:12.4f}{h['val_macro_auroc']:12.4f}"
          f"{h['val_macro_auprc']:12.4f}{h['val_macro_f1']:10.4f}{h['seconds']/60:8.1f}")
    A("")
    A("-" * 92)
    A("NOTES")
    A("-" * 92)
    A("  * Prototype subset: train is subsampled at 1 augmentation variant per record.")
    A("    AUROC/AUPRC and the RANKING of architectures transfer to the full corpus;")
    A("    thresholds, calibration and absolute scores do not.")
    A("  * 'accuracy' for a multi-label model is ambiguous. Both readings are given:")
    A("    exact-match (all 4 heads right on one ECG) and hamming (per label slot).")
    A("  * FPR = 1 - specificity; both are reported because you asked for both.")
    A("=" * 92)
    return "\n".join(L)


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    art = artifacts.RunArtifacts(args.out, args.model, resume=args.resume)
    try:
        print(f"config fingerprint: {C.fingerprint()}   (must match across all runs)")
        dinfo = device_info()
        print(f"device: {dinfo['device']} ({dinfo['vram_gb']} GB)")
        if device.type == "cpu":
            print("WARNING: no GPU. In Colab: Runtime > Change runtime type > GPU")

        mb = models.micro_batch(args.model, args.micro_batch)
        accum = models.accumulation_steps(args.model, args.micro_batch)
        print(f"batching: micro {mb} x accum {accum} = effective {C.EFFECTIVE_BATCH} "
              f"(identical for every model)")

        print("\nloading dataset ...")
        t0 = time.time()
        loaders = D.build_loaders(args.data, mb, args.workers)
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
        bench = engine.benchmark_inference(model, device, batch_size=mb,
                                           reps=10 if args.smoke else 30)
        print(f"  {comp['params_total_m']:.1f} M params | {comp['size_mb']:.1f} MB | "
              f"{bench['latency_ms_b1_mean']:.2f} ms/image (batch=1) | "
              f"{bench['throughput_ips_batched']:.1f} img/s batched")

        # ---- persist --------------------------------------------------------
        result = {
            "model": args.model,
            "fingerprint": C.fingerprint(),
            "controlled_settings": C.controlled_settings(),
            "model_info": minfo,
            "complexity": comp,
            "inference": bench,
            "data": {"root": str(Path(args.data).resolve()), **loaders["report"]},
            "pos_weight": loaders["pos_weight"],
            "lr_used": base_lr,
            "sweep": bool(args.sweep),
            "smoke": bool(args.smoke),
            "epochs_run": len(history),
            "best_val_monitor": float(best_val),
            "history": history,
            "val": val_res,
            "test": test_res,
            "test_per_source": per_src,
            "thresholds": thresholds,
            "train_seconds": elapsed,
            "env": {**dinfo, "torch": torch.__version__,
                    "python": platform.python_version()},
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        (art.dir / "result.json").write_text(json.dumps(result, indent=2, default=str))

        art.write_text("report.txt", build_report(args, result, val_res, test_res,
                                                  per_src, comp, bench, loaders, history))
        art.write_csv("metrics.csv",
                      metrics.flat_rows(val_res, "val", args.model)
                      + metrics.flat_rows(test_res, "test", args.model))
        if not args.smoke:
            art.save_predictions("val", loaders["val_df"], vy, vp, thresholds)
            art.save_predictions("test", loaders["test_df"].iloc[tidx].reset_index(drop=True),
                                 ty, tp, thresholds)

        print("\n" + (art.dir / "report.txt").read_text())

        # last.pt exists only to resume an interrupted run. This run finished, so
        # it is now ~16 bytes/param of dead weight sitting in Drive. best.pt has
        # the weights that matter.
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
