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
               f"fingerprint: {C.fingerprint()}\n")
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
        """Append one epoch immediately, so an interrupted run still has curves."""
        new = not self.history_path.exists()
        if self._hist_fields is None:
            self._hist_fields = list(row)
        with self.history_path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self._hist_fields, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(row)
            f.flush()

    # ------------------------------------------------------------------ config
    def snapshot_config(self) -> None:
        (self.dir / "config_snapshot.json").write_text(json.dumps({
            "fingerprint": C.fingerprint(),
            "controlled_settings": C.controlled_settings(),
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
    lines = ["CardioSentry - run summary",
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
                rows.append((d.name, status, t["macro_auroc"], t["macro_auprc"],
                             t["macro_f1"], t["macro_sensitivity"],
                             r.get("complexity", {}).get("params_total_m", float("nan")),
                             r.get("complexity", {}).get("size_mb", float("nan")),
                             r.get("inference", {}).get("latency_ms_b1_mean", float("nan")),
                             r.get("train_seconds", 0) / 60))
                continue
            except Exception:
                pass
        rows.append((d.name, status, *([float("nan")] * 8)))

    hdr = (f"{'model':18}{'status':9}{'mAUROC':>9}{'mAUPRC':>9}{'mF1':>8}"
           f"{'mSens':>8}{'params M':>10}{'size MB':>9}{'ms/img':>9}{'min':>8}")
    lines += [hdr, "-" * len(hdr)]
    for r in rows:
        lines.append(f"{r[0]:18}{r[1]:9}{r[2]:9.4f}{r[3]:9.4f}{r[4]:8.4f}"
                     f"{r[5]:8.4f}{r[6]:10.1f}{r[7]:9.1f}{r[8]:9.2f}{r[9]:8.1f}")
    lines += ["", "run `python compare.py --runs <this dir>` for the full comparison,",
              "which also verifies the runs were produced under identical conditions."]
    p = root / "_SUMMARY.txt"
    p.write_text("\n".join(lines))
    return p
