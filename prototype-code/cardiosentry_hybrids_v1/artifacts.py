"""
Run artifacts: everything worth keeping, written to disk as it happens.

Built for unattended overnight runs on Colab, so the guiding rule is:
NOTHING IS HELD IN MEMORY UNTIL THE END. Logs and per-epoch history are
flushed the moment they are produced, so a disconnect, an OOM at 4am or a
runtime reset still leaves a readable record of everything up to that point.

Layout under --out (point it at Drive):

    <out>/
      _SUMMARY.txt                 all models at a glance, rewritten per model
      comparison.txt / .csv        written by compare.py
      <model>/
        STATUS.txt                 RUNNING | DONE | FAILED  + timestamps
        train_log.txt              full console output, flushed live
        history.csv                one row per epoch, appended live
        report.txt                 human-readable final metrics
        metrics.csv                flat per-label + macro, for plotting
        result.json                machine-readable everything
        config_snapshot.json       controlled settings + fingerprint
        weights/best.pt            best-on-val weights + thresholds
        weights/last.pt            resume checkpoint
        predictions/val_predictions.csv
        predictions/test_predictions.csv
"""
from __future__ import annotations
import csv, json, sys, time, traceback
from pathlib import Path

import config as C


class Tee:
    """Duplicate stdout into a file, flushing every write.

    Buffered logs are the classic way to lose the last 200 lines exactly when
    they matter most - the crash. Flushing every line costs nothing here.
    """

    def __init__(self, path: Path, stream=None):
        self.file = open(path, "a", buffering=1, encoding="utf-8")
        self.stream = stream or sys.__stdout__

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
        self.file.write(data)
        self.file.flush()
        return len(data)

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def isatty(self):
        return False

    def close(self):
        try:
            self.file.close()
        except Exception:
            pass


class RunArtifacts:
    def __init__(self, out_root: str | Path, model: str, resume: bool = False):
        self.model = model
        self.root = Path(out_root)
        self.dir = self.root / model
        self.weights = self.dir / "weights"
        self.preds = self.dir / "predictions"
        for d in (self.dir, self.weights, self.preds):
            d.mkdir(parents=True, exist_ok=True)

        self.log_path = self.dir / "train_log.txt"
        self.history_path = self.dir / "history.csv"
        self.status_path = self.dir / "STATUS.txt"
        self.started = time.time()

        self._tee = Tee(self.log_path)
        sys.stdout = self._tee

        self._hist_fields: list[str] | None = None
        if not resume and self.history_path.exists():
            self.history_path.unlink()

        print("\n" + "=" * 78)
        print(f"RUN START  {model}   {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"artifacts -> {self.dir}")
        print("=" * 78)
        self.set_status("RUNNING")
        self.snapshot_config()

    # ------------------------------------------------------------------ status
    def set_status(self, state: str, note: str = "") -> None:
        """A one-glance answer to 'what happened while I was asleep?'"""
        el = time.time() - self.started
        txt = (f"status:   {state}\n"
               f"model:    {self.model}\n"
               f"started:  {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.started))}\n"
               f"updated:  {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
               f"elapsed:  {el/60:.1f} min\n"
               f"protocol fingerprint: {C.protocol_fingerprint()}\n")
        if note:
            txt += f"\n{note}\n"
        self.status_path.write_text(txt)

    def fail(self, exc: BaseException) -> None:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        print("\n" + "!" * 78)
        print(f"RUN FAILED: {type(exc).__name__}: {exc}")
        print("!" * 78)
        print(tb)
        self.set_status("FAILED", f"{type(exc).__name__}: {exc}\n\n{tb}")
        (self.dir / "ERROR.txt").write_text(tb)

    # ----------------------------------------------------------------- history
    def append_epoch(self, row: dict) -> None:
        """Append one epoch immediately, so an interrupted run still has curves.

        On RESUME the file already exists and was written by an earlier session.
        Its header is the authority, not this row's keys: appending a row whose
        key order differs would silently shift every value into the wrong column
        for the rest of the run, and the curve would look like a training
        anomaly rather than a bookkeeping bug. New keys added since (a metric
        introduced mid-project) are dropped by extrasaction="ignore" rather than
        corrupting the file - restart the run to pick them up.
        """
        new = not self.history_path.exists()
        if self._hist_fields is None:
            if new:
                self._hist_fields = list(row)
            else:
                with self.history_path.open(newline="") as f:
                    existing = next(csv.reader(f), None)
                self._hist_fields = existing or list(row)
                dropped = [k for k in row if k not in self._hist_fields]
                if dropped:
                    print(f"  note: history.csv predates {dropped} - those "
                          f"columns are omitted for this run. Delete "
                          f"{self.history_path.name} and restart to capture them.")
        with self.history_path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self._hist_fields, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(row)
            f.flush()

    # ------------------------------------------------------------------ config
    def snapshot_config(self) -> None:
        (self.dir / "config_snapshot.json").write_text(json.dumps({
            "protocol_fingerprint": C.protocol_fingerprint(),
            "input_fingerprint": C.input_fingerprint(),
            "protocol_settings": C.protocol_settings(),
            "input_settings": C.input_settings(),
        }, indent=2, default=str))

    # ------------------------------------------------------------------ output
    def write_text(self, name: str, text: str) -> Path:
        p = self.dir / name
        p.write_text(text, encoding="utf-8")
        return p

    def write_csv(self, name: str, rows: list[dict]) -> Path | None:
        if not rows:
            return None
        p = self.dir / name
        fields: list[str] = []
        for r in rows:
            for k in r:
                if k not in fields:
                    fields.append(k)
        with p.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        return p

    def save_predictions(self, split: str, df, y_true, y_prob, thresholds) -> Path:
        """Per-image probabilities. Lets you redo threshold analysis, error
        review or subgroup breakdowns later without retraining anything."""
        p = self.preds / f"{split}_predictions.csv"
        cols = ["record", "rel_path", "source", "split"]
        cols = [c for c in cols if c in df.columns]
        out = df[cols].copy()
        for k, lab in enumerate(C.LABELS):
            out[f"true_{lab}"] = y_true[:, k].astype(int)
            out[f"prob_{lab}"] = y_prob[:, k]
            out[f"pred_{lab}"] = (y_prob[:, k] >= thresholds[k]).astype(int)
        out.to_csv(p, index=False)
        return p

    def finalize(self, ok: bool = True, note: str = "") -> None:
        self.set_status("DONE" if ok else "FAILED", note)
        print(f"\nartifacts written to {self.dir}")
        for f in sorted(self.dir.rglob("*")):
            if f.is_file():
                print(f"   {f.relative_to(self.dir).as_posix():38} "
                      f"{f.stat().st_size/1024:9.1f} KB")
        sys.stdout = self._tee.stream
        self._tee.close()


# --------------------------------------------------------------- run summary
def write_summary(out_root: str | Path) -> Path:
    """Rewritten after every model, so a half-finished overnight batch is still
    readable in the morning without running anything."""
    root = Path(out_root)
    lines = ["CardioSentry hybrids - run summary",
             f"generated {time.strftime('%Y-%m-%d %H:%M:%S')}",
             f"location  {root}", ""]

    rows = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        status = "?"
        sp = d / "STATUS.txt"
        if sp.exists():
            first = sp.read_text().splitlines()[0]
            status = first.split(":", 1)[1].strip() if ":" in first else first
        rj = d / "result.json"
        if rj.exists():
            try:
                r = json.loads(rj.read_text())
                t = r["test"]
                views = "+".join(str(v) for v in r.get("model_info", {})
                                 .get("views", [])) or "?"
                per = [t["per_label"][l]["auprc"] for l in C.LABELS]
                # regressions against round 1, computed here so the morning
                # glance already answers "did anything break" without opening
                # compare.py
                try:
                    import baseline
                    worse = [l for l in C.LABELS
                             if t["per_label"][l]["auprc"]
                             - baseline.reference("auprc", l) < -0.02]
                    ss = (t["per_label"]["STEMI"]["sensitivity"]
                          - baseline.reference("sensitivity", "STEMI"))
                    if ss < -0.01:
                        worse.append("STEMI-sens")
                    flag = ",".join(worse) if worse else "-"
                except Exception:
                    flag = "?"
                # Focus label inside the primary database. The pooled column
                # next to it is inflated by database recognition on this corpus
                # (see subgroup.py), so the glance table carries both or it
                # misleads before anyone opens compare.py.
                try:
                    import subgroup
                    q = (r["test_per_source"][subgroup.PRIMARY_SOURCE]
                          ["per_label"][C.FOCUS_LABEL]["auprc"])
                except Exception:
                    q = float("nan")
                # LVH recall inside the AF+LVH subgroup - the question notebook
                # C exists to answer (frozen finding: resnet50_hires scored
                # 0.6429 here against 0.8722 on LVH-only). Computed from the
                # run's own frozen decisions; see af_lvh.py.
                try:
                    import af_lvh
                    af = af_lvh.score_run(d)
                    af_rec = (af["lvh_recall_af_lvh"] if af else float("nan"))
                except Exception:
                    af_rec = float("nan")
                rows.append((d.name, status, views, t["macro_auprc"], per, flag,
                             r.get("train_seconds", 0) / 60, q, af_rec))
                continue
            except Exception:
                pass
        rows.append((d.name, status, "?", float("nan"),
                     [float("nan")] * len(C.LABELS), "?", float("nan"),
                     float("nan"), float("nan")))

    try:
        import subgroup
        PS = subgroup.PRIMARY_SOURCE
    except Exception:
        PS = "PTBXL"
    foc_col = f"{C.FOCUS_LABEL}@{PS}"
    hdr = (f"{'model':22}{'status':9}{'views':>10}{'mAUPRC':>9}"
           + "".join(f"{l:>9}" for l in C.LABELS)
           + f"{foc_col:>13}{'A+Lrec':>9}{'min':>7}   regressed")
    lines += ["per-head test AUPRC. All four, on purpose - see the note below.", "",
              hdr, "-" * len(hdr)]
    for r in rows:
        q = r[7]
        af_rec = r[8]
        lines.append(f"{r[0]:22}{r[1]:9}{r[2]:>10}{r[3]:9.4f}"
                     + "".join(f"{v:9.4f}" for v in r[4])
                     + (f"{'':>13}" if q != q else f"{q:13.4f}")
                     + (f"{'':>9}" if af_rec != af_rec else f"{af_rec:9.4f}")
                     + f"{r[6]:7.1f}   {r[5]}")
    lines += ["",
              "WHY ALL FOUR HEADS ARE SHOWN, NOT JUST LVH:",
              "  This round exists to improve LVH, but macro averages dilute -- a",
              "  model that gains 0.04 on LVH and loses 0.04 on STEMI has an",
              "  unchanged macro AUPRC and looks like a clean win. It is not one.",
              "  An over-called LVH earns a follow-up echo; a missed STEMI does not.",
              "",
              "  'regressed' lists any head more than 0.02 AUPRC below round 1's",
              "  resnet50, plus STEMI-sens if sensitivity dropped by over 0.01.",
              "  '-' means nothing regressed.",
              "",
              f"  Round 1 resnet50 reference AUPRC: "
              + "  ".join(f"{l} {v:.4f}" for l, v in
                          zip(C.LABELS, [0.9560, 0.9777, 0.8805, 0.9534])),
              "",
              f"WHY THERE IS A SEPARATE {foc_col} COLUMN:",
              f"  {PS} carries 234 of the 255 {C.FOCUS_LABEL} positives; the STEMI",
              f"  database carries none, so a model that merely learns to recognise",
              f"  that database stops calling {C.FOCUS_LABEL} on its 146 images and its",
              f"  POOLED column rises without it reading a single QRS complex. That is",
              "  most of what 384->768 did in notebook A: pooled +0.0212, but only",
              f"  +0.0045 inside {PS}. The {foc_col} column is the one the",
              "  shortcut cannot reach. Rank on it, not on the column to its left.",
              "",
              f"  Notebook A's bar: resnet50_hires (768) = 0.9315 {foc_col}.",
              "",
              "WHY THERE IS AN 'A+Lrec' COLUMN:",
              "  The AF x LVH subgroup analysis (af_lvh_joint_v1) scored the",
              "  reference and found resnet50_hires detects AF but misses LVH",
              "  when the two coexist: LVH recall 0.6429 on the 28 AF+LVH cases",
              "  vs 0.8722 on LVH-only (Fisher one-sided p = 0.0038). This column",
              "  is every run's LVH recall inside those same 28 cases, from its",
              "  own frozen decisions. resnet50_vit@768 exists to raise it",
              "  without dropping the AF head.",
              "",
              "run `python compare.py --runs <this dir>` for the full comparison,",
              "which also verifies the runs shared the training protocol."]
    p = root / "_SUMMARY.txt"
    p.write_text("\n".join(lines))
    return p
