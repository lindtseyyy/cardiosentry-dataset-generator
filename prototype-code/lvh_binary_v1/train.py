#!/usr/bin/env python3
"""
Train the LVH-only binary ResNet-50@768 under the round-2 protocol.

    python train.py --model resnet50_lvh_binary --data /content/prototype_1152_768

Then `python compare.py --runs <dir> --reference <hybrid_v1 dir>` to tabulate
the binary run against the four-head reference. compare.py verifies the SHARED
protocol settings match the reference field by field before it will compare
them; the task (1 head vs 4) is the experimental variable and is reported
rather than enforced.

Colab-safe: checkpoints every epoch and resumes automatically after a disconnect.
"""
from __future__ import annotations
import argparse, json, os, platform, time
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
    p.add_argument("--model", default="resnet50_lvh_binary", choices=models.MODEL_NAMES)
    p.add_argument("--data", required=True,
                   help=f"prototype folder pre-resized to {C.LOAD_SIZE}px "
                        f"(the SAME prototype_1152_768 the four-head reference read)")
    p.add_argument("--out", default="runs", help="where to write runs/<model>/")
    p.add_argument("--epochs", type=int, default=None, help="override config.EPOCHS")
    p.add_argument("--lr", type=float, default=None,
                   help="override config.LR - BREAKS the comparison with the "
                        "four-head reference unless you also re-run it at that LR")
    p.add_argument("--micro-batch", type=int, default=None,
                   help="VRAM accommodation only; the reference used 8. Effective "
                        "batch stays 32, but BatchNorm normalises over the micro-"
                        "batch, so changing this breaks comparability.")
    p.add_argument("--workers", type=int, default=0,
                   help="0 = auto. Decoding is CPU-bound, so more workers than "
                        "cores buys nothing. Pass a number to override.")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--no-channels-last", action="store_true",
                   help="disable NHWC. Layout only - use it to check the "
                        "optimisation is not perturbing anything, not routinely.")
    p.add_argument("--smoke", action="store_true",
                   help="20 train / 10 eval batches, 1 epoch - proves the whole "
                        "path works in ~2 min before you commit the run")
    p.add_argument("--sweep", action="store_true",
                   help="equal-budget LR search over config.LR_GRID. The reference "
                        "did NOT sweep (it used 1e-4), so a swept run is not the "
                        "controlled comparison - use it only for diagnostics.")
    p.add_argument("--keep-last", action="store_true",
                   help="keep last.pt after the run finishes (it is only needed "
                        "to RESUME and is deleted on success by default)")
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

    model, minfo = models.build(args.model, pretrained=not args.no_pretrained)
    model.to(device)

    use_cl = bool(minfo.get("channels_last")) and not args.no_channels_last \
             and device.type == "cuda"
    if use_cl:
        model = model.to(memory_format=torch.channels_last)
    model.channels_last = use_cl
    minfo["channels_last"] = use_cl

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
        if blob.get("protocol_fingerprint") != C.protocol_fingerprint():
            print(f"  checkpoint protocol {blob.get('protocol_fingerprint')} != "
                  f"current {C.protocol_fingerprint()} - starting fresh")
        elif blob.get("task_fingerprint") != C.task_fingerprint():
            print(f"  checkpoint task {blob.get('task_fingerprint')} != current "
                  f"{C.task_fingerprint()} - starting fresh")
        elif blob.get("views") not in (None, list(model.views)):
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
               "val_macro_fpr": val["macro_fpr"],
               "val_exact_match_acc": val["exact_match_accuracy"],
               "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
        for _l in C.LABELS:
            row[f"val_{_l}_auprc"] = val["per_label"][_l]["auprc"]
            row[f"val_{_l}_sens"] = val["per_label"][_l]["sensitivity"]
            row[f"val_{_l}_fpr"] = val["per_label"][_l]["fpr"]
        history.append(row)
        if art is not None:
            art.append_epoch(row)
            art.set_status("RUNNING", f"epoch {ep+1}/{epochs}  "
                                      f"best val {C.MONITOR} {max(best_val, score):.4f}")
        if not quiet:
            print(f"    train loss {tr['loss']:.4f}  |  val LVH AUPRC "
                  f"{val['per_label']['LVH']['auprc']:.4f}  |  "
                  + f"({tr['seconds']/60:.1f} min)", flush=True)

        improved = score > best_val
        if improved:
            best_val, bad = score, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save({"model": best_state, "thresholds": thr, "epoch": ep,
                        "val": val, "views": list(model.views),
                        "protocol_fingerprint": C.protocol_fingerprint(),
                        "task_fingerprint": C.task_fingerprint()},
                       ckpt_dir / f"best{tag_suffix}.pt")
        else:
            bad += 1

        if not args.smoke:
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict(), "epoch": ep, "history": history,
                        "best_val": best_val, "best_state": best_state, "bad": bad,
                        "views": list(model.views),
                        "protocol_fingerprint": C.protocol_fingerprint(),
                        "task_fingerprint": C.task_fingerprint(),
                        "data_root": str(Path(args.data).resolve())}, last_ckpt)

        if bad >= C.EARLY_STOP_PATIENCE and not args.smoke:
            print(f"    early stop: no {C.MONITOR} improvement in "
                  f"{C.EARLY_STOP_PATIENCE} epochs")
            break

    return model, best_state, history, best_val, minfo


def px_per_mm(views) -> dict:
    """Measurement precision at each view size. 150 DPI source, 1686x1311 sheet."""
    src_dpi_px_per_mm = 150 / 25.4          # 5.906
    out = {}
    for v in views:
        vert = src_dpi_px_per_mm * (v / 1311)
        horiz = src_dpi_px_per_mm * (v / 1686)
        out[str(v)] = {"vertical": vert, "horizontal": horiz, "mv_px": vert * 10}
    return out


def build_report(result, val_res, test_res, per_src, comp, bench, history, extra) -> str:
    """The file you read when the run finishes. Everything, in plain text."""
    import baseline
    L = []
    A = L.append
    A("=" * 92)
    A(f"LVH binary - negative-transfer test  |  {result['model']}")
    A("=" * 92)
    A(f"purpose           {result['model_info']['why']}")
    A(f"finished          {result['finished_utc']}")
    A(f"training time     {result['train_seconds']/60:.1f} min over "
      f"{result['epochs_run']} epoch(s)")
    A(f"protocol fp       {result['protocol_fingerprint']}   (full protocol, task included)")
    A(f"shared fp         {result['shared_fingerprint']}   <- must equal the reference's")
    A(f"task fp           {result['task_fingerprint']}   <- EXPECTED to differ; it is the variable")
    A(f"input views       {result['model_info']['views']}   (fp "
      f"{result['model_info']['input_fingerprint']})")
    A(f"dataset           {result['data']['root']}")
    A(f"                  {result['data']['rows']:,} images | "
      f"{result['data']['records']:,} records | {result['data']['patients']:,} patients")
    A(f"backbone tags     {result['model_info']['tags']}")
    A(f"pretraining       {result['model_info']['pretrain']}")
    A(f"learning rate     {result['lr_used']:.1e}"
      + ("  (equal-budget sweep)" if result['sweep'] else ""))
    A("")
    A("-" * 92)
    A("THE FAIRNESS CONTRACT - what was held constant against the reference")
    A("-" * 92)
    A(f"  shared protocol fingerprint {result['shared_fingerprint']}"
      f"  vs reference {baseline.REFERENCE_SHARED_FINGERPRINT}")
    A(f"  {'MATCH' if result['shared_fingerprint'] == baseline.REFERENCE_SHARED_FINGERPRINT else 'MISMATCH'}"
      f" - same seed, norm, aug, optimiser, effective batch 32, epochs,"
      f" monitor name, early stopping, threshold rule")
    A(f"  pos_weight LVH  {result['pos_weight'][0]:.4f}   (reference used "
      f"{baseline.REFERENCE_POS_WEIGHT_LVH:.4f})")
    A(f"  micro-batch     {result['micro_batch']}   (reference used "
      f"{baseline.REFERENCE_META['micro_batch']})")
    A("  ONE declared difference: 4 heads -> 1. Head structure unchanged except")
    A("  the final width (512x4 -> 512x1). Monitor name unchanged; over one head")
    A("  'macro_auprc' IS the LVH head's val AUPRC.")
    A("")
    A("-" * 92)
    A("MEASUREMENT PRECISION AT THIS INPUT SIZE")
    A("-" * 92)
    for v in result["model_info"]["views"]:
        A(f"  {v:>4}px view:  {extra['px_per_mm'][str(v)]['vertical']:.2f} px/mm "
          f"vertical ({extra['px_per_mm'][str(v)]['mv_px']:.1f} px per mV)   "
          f"{extra['px_per_mm'][str(v)]['horizontal']:.2f} px/mm horizontal")
    A("")
    A("-" * 92)
    A("MODEL COMPLEXITY  (must match the reference - same backbone, head 512x1)")
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
    A(f"  batched (batch={bench['batch_size']})        "
      f"{bench['ms_per_image_batched']:8.2f} ms/image")
    A("")
    A("-" * 92)
    A("VALIDATION  (threshold tuned here, then frozen)")
    A("-" * 92)
    A(metrics.format_report(val_res))
    A("")
    A("-" * 92)
    A("TEST  (val-tuned threshold, frozen before test - no tuning on test)")
    A("-" * 92)
    A(metrics.format_report(test_res))
    A("")
    A("-" * 92)
    A("LVH FOCUS  -  the number this round exists to move")
    A("-" * 92)
    f = test_res["per_label"]["LVH"]
    A(f"  AUPRC        {f['auprc']:.4f}          AUROC       {f['auroc']:.4f}")
    A(f"  sensitivity  {f['sensitivity']:.4f}          specificity {f['specificity']:.4f}")
    A(f"  precision    {f['precision']:.4f}          F1          {f['f1']:.4f}")
    A(f"  FPR          {f['fpr']:.4f}          threshold   {f['threshold']:.2f}")
    A(f"  TP {f['tp']:>4}   FP {f['fp']:>4}   FN {f['fn']:>4}   TN {f['tn']:>4}"
      f"     ({f['support']} positives of {f['n']})")
    A("")
    A("  Four-head reference (resnet50_hires@768), pooled test:")
    r = baseline.REFERENCE_LVH["pooled"]
    A(f"    AUPRC {r['auprc']:.4f}  AUROC {r['auroc']:.4f}  sens {r['sensitivity']:.4f}"
      f"  spec {r['specificity']:.4f}  prec {r['precision']:.4f}  F1 {r['f1']:.4f}"
      f"  FPR {r['fpr']:.4f}")
    A(f"    TP {r['tp']}  FP {r['fp']}  FN {r['fn']}  TN {r['tn']}")
    A("  PTB-XL reference (the primary read - the source shortcut cannot reach it):")
    q = baseline.REFERENCE_LVH["PTBXL"]
    A(f"    AUPRC {q['auprc']:.4f}  AUROC {q['auroc']:.4f}  sens {q['sensitivity']:.4f}"
      f"  spec {q['specificity']:.4f}  prec {q['precision']:.4f}  F1 {q['f1']:.4f}"
      f"  FPR {q['fpr']:.4f}")
    A("")
    if per_src:
        A("-" * 92)
        A("SOURCE CONFOUND CHECK")
        A("-" * 92)
        A(metrics.format_per_source(per_src))
        A("")
        A("  The STEMI database contributes 146 images and ZERO LVH positives, so")
        A("  pooled LVH can be inflated by database recognition. The PTB-XL row")
        A("  above is the one that cannot be - rank on it.")
        A("")
    A("-" * 92)
    A("PER-EPOCH HISTORY")
    A("-" * 92)
    A(f"  {'ep':>3}{'train_loss':>12}{'val AUPRC':>12}{'val AUROC':>12}"
      f"{'val sens':>11}{'val FPR':>10}{'min':>8}")
    for h in history:
        A(f"  {h['epoch']:>3}{h['train_loss']:12.4f}{h['val_macro_auprc']:12.4f}"
          f"{h['val_macro_auroc']:12.4f}{h['val_macro_sensitivity']:11.4f}"
          f"{h['val_macro_fpr']:10.4f}{h['seconds']/60:8.1f}")
    A("")
    A("-" * 92)
    A("NOTES")
    A("-" * 92)
    A("  * Prototype subset: train is subsampled at 1 augmentation variant per")
    A("    record. AUROC/AUPRC and the comparison transfer to the full corpus;")
    A("    thresholds, calibration and absolute scores do not.")
    A("  * The head is the SAME 2-layer MLP as the four-head reference, only the")
    A("    final width changed (512x4 -> 512x1). That width is the experiment.")
    A("  * Monitor 'macro_auprc' over one head equals the LVH head's val AUPRC;")
    A("    the reference selected its checkpoint on val macro AUPRC over four")
    A("    heads. This is the one selection-rule consequence of the task change,")
    A("    and it is declared, not hidden.")
    A("=" * 92)
    return "\n".join(L)


def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    art = artifacts.RunArtifacts(args.out, args.model, resume=args.resume)
    try:
        print(f"protocol fingerprint: {C.protocol_fingerprint()}   "
              f"(full protocol, task included)")
        print(f"shared   fingerprint: {C.shared_fingerprint()}   "
              f"(must match the four-head reference)")
        print(f"task     fingerprint: {C.task_fingerprint()}   "
              f"(this is the experimental variable)")
        views = models.views(args.model)
        print(f"input views: {list(views)}")
        dinfo = device_info()
        print(f"device: {dinfo['device']} ({dinfo['vram_gb']} GB)")
        if device.type == "cpu":
            print("WARNING: no GPU. In Colab: Runtime > Change runtime type > GPU")

        mb = models.micro_batch(args.model, args.micro_batch)
        accum = models.accumulation_steps(args.model, args.micro_batch)
        print(f"batching: micro {mb} x accum {accum} = effective "
              f"{C.EFFECTIVE_BATCH} (reference used micro 8)")

        print("\nloading dataset ...")
        t0 = time.time()
        n_workers = D.resolve_workers(args.workers)
        print(f"dataloader: {n_workers} worker(s) on {os.cpu_count()} CPU(s)")
        loaders = D.build_loaders(args.data, mb, views, n_workers)
        print(manifest.describe(loaders["df"]))
        print(f"  integrity checks passed in {time.time()-t0:.1f}s")
        print(f"  pos_weight LVH = {loaders['pos_weight'][0]:.4f}")

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
            print("  NOTE: the reference did NOT sweep. A swept run is not the")
            print("  controlled comparison - it is a diagnostic.")
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
        print(metrics.format_report(val_res, "VALIDATION (threshold tuned here)"))

        tp, ty, tidx = engine.predict(model, loaders["test"], device, max_batches=max_eb)
        test_res = metrics.evaluate(ty, tp, thresholds)
        print()
        print(metrics.format_report(test_res, "TEST (val-tuned threshold, frozen)"))

        per_src = {}
        if C.REPORT_PER_SOURCE:
            src = loaders["test_df"].iloc[tidx]["source"].tolist()
            per_src = metrics.evaluate_per_source(ty, tp, src, thresholds)
            print()
            print(metrics.format_per_source(per_src))

        # ---- complexity + speed --------------------------------------------
        print("\nbenchmarking inference ...")
        comp = engine.model_complexity(model)
        bench = engine.benchmark_inference(model, device, views, batch_size=mb,
                                           reps=10 if args.smoke else 30)
        print(f"  {comp['params_total_m']:.1f} M params | {comp['size_mb']:.1f} MB | "
              f"{bench['latency_ms_b1_mean']:.2f} ms/image (batch=1) | "
              f"{bench['throughput_ips_batched']:.1f} img/s batched")

        extra = {"px_per_mm": px_per_mm(views)}

        # ---- persist --------------------------------------------------------
        result = {
            "model": args.model,
            "task": {"labels": C.LABELS, "num_classes": C.NUM_CLASSES},
            "protocol_fingerprint": C.protocol_fingerprint(),
            "task_fingerprint": C.task_fingerprint(),
            "shared_fingerprint": C.shared_fingerprint(),
            "input_fingerprint": minfo["input_fingerprint"],
            "protocol_settings": C.protocol_settings(),
            "task_settings": C.task_settings(),
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
            "test_per_source": per_src,
            "thresholds": thresholds,
            "extra": extra,
            "train_seconds": elapsed,
            "env": {**dinfo, "torch": torch.__version__,
                    "python": platform.python_version()},
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        (art.dir / "result.json").write_text(json.dumps(result, indent=2, default=str))

        art.write_text("report.txt", build_report(result, val_res, test_res, per_src,
                                                  comp, bench, history, extra))
        art.write_csv("metrics.csv",
                      metrics.flat_rows(val_res, "val", args.model)
                      + metrics.flat_rows(test_res, "test", args.model))
        if not args.smoke:
            art.save_predictions("val", loaders["val_df"], vy, vp, thresholds)
            art.save_predictions("test",
                                 loaders["test_df"].iloc[tidx].reset_index(drop=True),
                                 ty, tp, thresholds)

        print("\n" + (art.dir / "report.txt").read_text())

        # last.pt exists only to resume an interrupted run.
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
