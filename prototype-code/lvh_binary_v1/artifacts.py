"""
Run artifacts: everything worth keeping, written to disk as it happens.

Same rules as the round-2 harness: NOTHING IS HELD IN MEMORY UNTIL THE END.
Logs and per-epoch history are flushed the moment they are produced, so a
disconnect, an OOM at 4am or a runtime reset still leaves a readable record.

Layout under --out (point it at Drive):

    <out>/
      _SUMMARY.txt                 the run at a glance, delta vs the reference
      comparison.txt               written by compare.py
      <model>/
        STATUS.txt                 RUNNING | DONE | FAILED  + timestamps
        train_log.txt              full console output, flushed live
        history.csv                one row per epoch, appended live
        report.txt                 human-readable final metrics
        metrics.csv                flat per-label rows, for plotting
        result.json                machine-readable everything
        config_snapshot.json       controlled settings + fingerprints
        weights/best.pt            best-on-val weights + frozen threshold
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
        el = time.time() - self.started
        txt = (f"status:   {state}\n"
               f"model:    {self.model}\n"
               f"started:  {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.started))}\n"
               f"updated:  {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
               f"elapsed:  {el/60:.1f} min\n"
               f"protocol fingerprint: {C.protocol_fingerprint()}\n"
               f"task fingerprint:     {C.task_fingerprint()}\n")
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

        On RESUME the existing header is the authority, not this row's keys -
        see the round-2 harness for the reasoning; the code is the same.
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
            "task_fingerprint": C.task_fingerprint(),
            "shared_fingerprint": C.shared_fingerprint(),
            "input_fingerprint": C.input_fingerprint(),
            "protocol_settings": C.protocol_settings(),
            "task_settings": C.task_settings(),
            "shared_settings": C.shared_settings(),
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
        review or the paired bootstrap later without retraining anything."""
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
    """Rewritten after the run, with the delta against the frozen four-head
    reference computed here so the morning glance already answers the round's
    question."""
    import baseline
    root = Path(out_root)
    lines = ["LVH binary - negative-transfer test - run summary",
             f"generated {time.strftime('%Y-%m-%d %H:%M:%S')}",
             f"location  {root}", "",
             f"REFERENCE: four-head {baseline.REFERENCE_MODEL}@768 (round 2)",
             f"  pooled LVH AUPRC {baseline.REFERENCE_LVH['pooled']['auprc']:.4f}"
             f"   PTB-XL LVH AUPRC {baseline.REFERENCE_LVH['PTBXL']['auprc']:.4f}",
             f"  PTB-XL LVH: sens {baseline.REFERENCE_LVH['PTBXL']['sensitivity']:.4f}"
             f"  FPR {baseline.REFERENCE_LVH['PTBXL']['fpr']:.4f}",
             "", "per-run test LVH metrics (delta vs the reference):", ""]

    hdr = (f"{'model':24}{'status':9}{'task':22}{'pooled':>9}"
           f"{'PTBXL':>9}{'dAUPRC':>9}{'dPTBXL':>9}{'min':>7}")
    lines += [hdr, "-" * len(hdr)]
    any_run = False
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        sp = d / "STATUS.txt"
        status = "?"
        if sp.exists():
            first = sp.read_text().splitlines()[0]
            status = first.split(":", 1)[1].strip() if ":" in first else first
        rj = d / "result.json"
        if not rj.exists():
            lines.append(f"{d.name:24}{status:9}")
            continue
        try:
            import json
            r = json.loads(rj.read_text())
            t = r["test"]
            lvh = t["per_label"][C.FOCUS_LABEL]
            q = r.get("test_per_source", {}).get("PTBXL", {}).get(
                "per_label", {}).get(C.FOCUS_LABEL, {})
            ptb = q.get("auprc", float("nan"))
            dp = lvh["auprc"] - baseline.REFERENCE_LVH["pooled"]["auprc"]
            dq = ptb - baseline.REFERENCE_LVH["PTBXL"]["auprc"]
            lines.append(f"{d.name:24}{status:9}{'LVH-only binary':22}"
                         f"{lvh['auprc']:9.4f}"
                         + (f"{'':>9}" if ptb != ptb else f"{ptb:9.4f}")
                         + f"{dp:+9.4f}{dq:+9.4f}"
                         + f"{r.get('train_seconds', 0)/60:7.1f}")
            any_run = True
        except Exception:
            lines.append(f"{d.name:24}{status:9}")
    if not any_run:
        lines += ["  (no finished run yet)"]
    lines += ["",
              "HOW TO READ THE DELTAS:",
              "  +dPTBXL means the LVH-only model beats the four-head reference",
              "  inside PTB-XL - evidence FOR negative transfer from the",
              "  multi-label objective. The pooled column is shown for context,",
              "  but the STEMI database contributes 146 LVH-negative images, so",
              "  pooled deltas can be database recognition, not ECG reading.",
              "",
              "run `python compare.py --runs <this dir>` for the full",
              "comparison, the paired bootstrap CIs and the verdict."]
    p = root / "_SUMMARY.txt"
    p.write_text("\n".join(lines))
    return p
