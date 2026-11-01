"""CPU validation of the CAM-comparison notebook, `xai_cam.py` and `cpu_timing.py`.

    ../../.venv/bin/python validate_notebook.py

Part 1 checks the method code on a tiny timm ConvNeXt (convnext_atto):
  - the model split equals the model
  - Grad-CAM / Grad-CAM++ equal pytorch-grad-cam (when it is importable;
    CS_GRADCAM_PATH may point at a folder that holds it)
  - Score-CAM does not depend on the batch size and equals a per-channel loop
  - pixel ranks, the curve endpoints, batch-size invariance of the curves, the AUC
  - an oracle map beats a random map on a toy model whose score lives in one corner

Part 2 runs every notebook code cell on CPU against a temporary "Drive": a tiny
ConvNeXt checkpoint (raw + EMA, EMA selected), synthetic sheets in a real TAR
with a SHA-256 sidecar, and test_predictions.csv scored by that EMA model. It
checks:
  - a full run
  - a rerun on an empty runtime with the big archive gone (Drive image cache +
    resume skip every image)
  - a changed design refused under the same run name
  - a parity failure when the reference probabilities are wrong
  - cpu_timing.py end to end
"""
import contextlib
import hashlib
import io
import json
import os
import shutil
import sys
import tarfile
import tempfile
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import timm
from PIL import Image

HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "CardioSentry_XAI_CAM_Comparison_Clean12_Colab.ipynb"
sys.path.insert(0, str(HERE))
if os.environ.get("CS_GRADCAM_PATH"):
    sys.path.insert(0, os.environ["CS_GRADCAM_PATH"])
import xai_cam as X  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")

RESULTS = []


def check(name, condition, detail=""):
    RESULTS.append((name, bool(condition), detail))
    print(f"{'PASS' if condition else 'FAIL'}  {name}{('  ' + detail) if detail else ''}", flush=True)


# ============================================================= part 1: methods
def part1():
    torch.manual_seed(0)
    model = timm.create_model("convnext_atto", pretrained=False, num_classes=12).eval()
    split = X.ModelSplit(model)
    x = torch.randn(1, 3, 96, 128)
    with torch.no_grad():
        check("split.head(split.features(x)) == model(x)",
              torch.allclose(split.head(split.features(x)), model(x), atol=1e-6))
    a = X.activations(split, x)
    maps, logits = X.gradient_cams(split, a, [2, 9])
    check("gradient_cams returns both methods for both labels",
          all(set(maps[m]) == {2, 9} for m in X.GRADIENT_METHODS))

    # Grad-CAM from first principles: full-graph hook on stages[3].
    store = {}
    handle = model.stages[3].register_forward_hook(lambda m, i, o: store.setdefault("a", o))
    xg = x.clone()
    out = model(xg)
    handle.remove()
    (grad,) = torch.autograd.grad(out[0, 2], store["a"])
    ref = torch.relu((grad[0].mean(dim=(1, 2))[:, None, None] * store["a"][0].detach()).sum(0))
    check("Grad-CAM equals a full-graph reference", torch.allclose(ref, maps["gradcam"][2], atol=1e-5))

    try:
        from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        for name, cls in (("gradcam", GradCAM), ("gradcampp", GradCAMPlusPlus)):
            for k in (2, 9):
                with cls(model=model, target_layers=[model.stages[-1]]) as cam:
                    theirs = cam(input_tensor=x.clone().requires_grad_(True),
                                 targets=[ClassifierOutputTarget(k)])[0]
                mine = X.upsample(maps[name][k], (96, 128)).numpy()
                mine = (mine - mine.min()) / (mine.max() - mine.min() + 1e-7)
                r = np.corrcoef(mine.ravel(), theirs.ravel())[0, 1]
                check(f"{name} label {k} matches pytorch-grad-cam", r > 0.99999, f"r={r:.6f}")
    except ImportError:
        print("SKIP  pytorch-grad-cam not importable (set CS_GRADCAM_PATH)")

    s1, f1 = X.scorecam_channel_scores(split, x, a, 1)
    s7, f7 = X.scorecam_channel_scores(split, x, a, 7)
    check("Score-CAM scores do not depend on the batch size",
          torch.allclose(s1, s7, atol=1e-5) and f1 == f7)
    naive = []
    with torch.no_grad():
        for c in range(a.shape[1]):
            up = torch.nn.functional.interpolate(a[:, c:c + 1], size=(96, 128), mode="bilinear",
                                                 align_corners=False)[0, 0]
            m = (up - up.min()) / (up.max() - up.min())
            naive.append(torch.sigmoid(model(x * m)[0]))
    check("Score-CAM scores equal a per-channel loop", torch.allclose(s1, torch.stack(naive), atol=1e-5))
    a_const = a.clone()
    a_const[:, :5] = 1.0
    _, f_const = X.scorecam_channel_scores(split, x, a_const, 16)
    check("constant channels are skipped", f_const == a.shape[1] - 5, f"{f_const} forwards")
    sc = X.scorecam_maps(a, s1, [2])
    check("Score-CAM map is ReLU(sum w A)",
          torch.allclose(sc[2], torch.relu(torch.einsum("c,chw->hw", s1[:, 2], a[0])), atol=1e-6))

    sal = torch.tensor([[0.2, 0.9, 0.9], [0.0, 0.0, 0.5]])
    rank = X.pixel_rank(sal, 3)
    flat_rank = rank.flatten().tolist()
    check("pixel_rank is a permutation", sorted(flat_rank) == list(range(6)))
    order = rank.flatten().argsort()
    check("pixel_rank follows descending saliency",
          all(sal.flatten()[order[i]] >= sal.flatten()[order[i + 1]] for i in range(5)))
    check("pixel_rank tie-break is seeded", torch.equal(rank, X.pixel_rank(sal, 3)))

    image01 = torch.rand(3, 96, 128)
    baseline = X.make_baseline(image01, "paper", 4)
    mean = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
    rank = X.pixel_rank(torch.rand(96, 128), 1)
    fr, de, ins = X.deletion_insertion(model, image01, baseline, rank, 4, 10, 3, mean, std)
    fr2, de2, ins2 = X.deletion_insertion(model, image01, baseline, rank, 4, 10, 11, mean, std)
    with torch.no_grad():
        p_img = torch.sigmoid(model(((image01 - mean) / std)[None]))[0, 4].item()
        p_base = torch.sigmoid(model(((baseline - mean) / std)[None]))[0, 4].item()
    check("curves: fractions 0..1 in steps + 1 points",
          len(fr) == 11 and fr[0] == 0 and fr[-1] == 1 and np.all(np.diff(fr) > 0))
    check("deletion starts at p(image), ends at p(baseline)",
          abs(de[0] - p_img) < 1e-5 and abs(de[-1] - p_base) < 1e-5)
    check("insertion starts at p(baseline), ends at p(image)",
          abs(ins[0] - p_base) < 1e-5 and abs(ins[-1] - p_img) < 1e-5)
    _, de3, ins3 = X.deletion_insertion(X.FixedBatch(model, 4), image01, baseline, rank, 4, 10, 4, mean, std)
    check("curves through FixedBatch equal eager",
          np.allclose(de, de3, atol=1e-5) and np.allclose(ins, ins3, atol=1e-5))
    check("curves do not depend on the batch size",
          np.allclose(de, de2, atol=1e-5) and np.allclose(ins, ins2, atol=1e-5))
    check("AUC of a constant and of the identity",
          abs(X.curve_auc([0, .5, 1], [.3, .3, .3]) - .3) < 1e-12
          and abs(X.curve_auc(np.linspace(0, 1, 11), np.linspace(0, 1, 11)) - .5) < 1e-12)
    paper = torch.ones(3, 20, 30) * torch.tensor([0.9, 0.8, 0.7]).view(3, 1, 1)
    paper[:, :, 10:] = 0.1
    colour, spread = X.paper_colour(paper, 4)
    check("paper colour comes from the side pad",
          torch.allclose(colour, torch.tensor([0.9, 0.8, 0.7])) and spread < 1e-6)
    check("blur and zero baselines have the image's shape",
          X.make_baseline(image01, "blur", 4, 2.0).shape == image01.shape
          and float(X.make_baseline(image01, "zero").abs().sum()) == 0.0)

    class Corner(torch.nn.Module):
        """Logit = 2 x mean of the (normalized) top-left 16 x 16 corner - 2.5 (p 0.11 .. 0.91)."""
        def forward(self, x):
            return (2 * x[:, :, :16, :16].mean(dim=(1, 2, 3)) - 2.5)[:, None].repeat(1, 3)

    toy_image = torch.zeros(3, 64, 64)
    toy_image[:, :16, :16] = 1.0
    toy_baseline = torch.full_like(toy_image, 0.5)
    oracle = torch.zeros(64, 64)
    oracle[:16, :16] = 1.0
    results = {}
    for name, sal in (("oracle", oracle), ("random", torch.rand(64, 64, generator=torch.Generator().manual_seed(0)))):
        fr, de, ins = X.deletion_insertion(Corner(), toy_image, toy_baseline, X.pixel_rank(sal, 0), 0,
                                           32, 8, mean, std)
        results[name] = (X.curve_auc(fr, de), X.curve_auc(fr, ins))
    check("oracle map: lower deletion AUC than random",
          results["oracle"][0] < results["random"][0] - 0.2, str(results))
    check("oracle map: higher insertion AUC than random", results["oracle"][1] > results["random"][1] + 0.2)

    fixed = X.FixedBatch(model, 8)
    batch = torch.randn(5, 3, 96, 128)
    with torch.no_grad():
        check("FixedBatch padding leaves the real rows' outputs unchanged",
              torch.allclose(fixed(batch), model(batch), atol=1e-5) and fixed(batch).shape[0] == 5)
    s_fixed, _ = X.scorecam_channel_scores(split, x, a, 7, forward=X.FixedBatch(model, 7))
    check("Score-CAM through FixedBatch equals eager", torch.allclose(s_fixed, s1, atol=1e-5))

    t, (cam, fw) = X.timed(lambda: X.explain(split, x, 3, "scorecam", None, 64), "cpu")
    check("explain(scorecam) returns an input-size map and 1 + C forwards",
          tuple(cam.shape) == (96, 128) and fw == 1 + a.shape[1], f"{fw} forwards, {t:.2f} s")
    _, (cam, fw) = X.timed(lambda: X.explain(split, x, 3, "gradcampp"), "cpu")
    check("explain(gradcampp) returns an input-size map and 2 forwards", tuple(cam.shape) == (96, 128) and fw == 2)


# ============================================================== part 2: notebook
LABELS = ["NORM", "SB", "STACH", "LBBB", "LQT", "AF", "AFL", "3AVB", "2AVB", "1AVB", "STEMI", "STE"]
HW = (96, 128)
PAD = 4
TOPDIR = "clean_ecg_signal_images_clean12"


def sheet(seed):
    rng = np.random.default_rng(seed)
    pixels = np.full((HW[0], HW[1], 3), (250, 238, 236), dtype=np.uint8)
    inner = pixels[:, PAD:HW[1] - PAD]
    inner[::8] = (240, 170, 170)
    inner[:, ::8] = (240, 170, 170)
    for row in range(4):
        base = 12 + 22 * row
        ys = base + (rng.normal(0, 3, HW[1] - 2 * PAD)).astype(int).clip(-8, 8)
        for col, y in enumerate(ys):
            inner[max(0, y - 1):y + 1, col] = (20, 20, 20)
    return Image.fromarray(pixels)


def build_fake_drive(root):
    drive = root / "drive" / "CardioSentry"
    run_dir = drive / "new_runs" / "convnextv1_base_clean12_a100_seed42"
    dataset_dir = drive / "new_dataset"
    run_dir.mkdir(parents=True)
    dataset_dir.mkdir(parents=True)

    torch.manual_seed(1)
    raw = timm.create_model("convnext_atto", pretrained=False, num_classes=12).eval()
    ema = timm.create_model("convnext_atto", pretrained=False, num_classes=12).eval()
    with torch.no_grad():
        for p in ema.parameters():
            p.add_(0.01 * torch.randn_like(p))
        ema.head.fc.bias.add_(torch.linspace(-1, 1, 12))
    torch.save({"model": raw.state_dict(), "model_ema": ema.state_dict(),
                "selected_weight_source": "ema", "epoch": 6}, run_dir / "best.pt")

    records = [f"I000{1 if i < 25 else 6}/WFDB/S0001/2006/11/rec_{i:03d}.hea" for i in range(40)]
    staging = root / "staging" / TOPDIR
    for i, rec in enumerate(records):
        path = staging / "images" / Path(rec).with_suffix(".png")
        path.parent.mkdir(parents=True, exist_ok=True)
        sheet(i).save(path)
    (staging / "manifest.csv").write_text("image_path\n")
    tar_path = dataset_dir / "clean_ecg_signal_images_clean12.tar"
    with tarfile.open(tar_path, "w") as archive:
        archive.add(staging, arcname=TOPDIR)
    digest = hashlib.sha256(tar_path.read_bytes()).hexdigest()
    (dataset_dir / "clean_ecg_signal_images_clean12.tar.sha256").write_text(
        f"{digest}  clean_ecg_signal_images_clean12.tar\n")

    mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    test_records = records[:30]
    images = torch.stack([torch.from_numpy(np.asarray(sheet(i)).copy()).permute(2, 0, 1).float() / 255
                          for i in range(30)])
    with torch.no_grad():
        probs = torch.sigmoid(ema((images - mean) / std)).numpy()
    rng = np.random.default_rng(7)
    frame = pd.DataFrame({"row_index": range(30), "record_id": test_records, "split": "test"})
    for j, label in enumerate(LABELS):
        truth = (rng.random(30) < 0.3).astype(int)
        truth[:3] = 1
        tuned = truth.copy()
        tuned[3:5] = 0                       # some false negatives are never chosen
        frame[f"true_{label}"] = truth
        frame[f"prob_{label}"] = probs[:, j]
        frame[f"pred_0_5_{label}"] = (probs[:, j] >= 0.5).astype(int)
        frame[f"pred_validation_tuned_{label}"] = tuned
    frame.to_csv(run_dir / "test_predictions.csv", index=False)
    return drive, run_dir, tar_path


def notebook_cells():
    nb = json.loads(NOTEBOOK.read_text())
    cells = []
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        source = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("%"))
        cells.append(source)
    return cells


def run_notebook(env):
    saved = dict(os.environ)
    os.environ.update(env)
    namespace = {"__name__": "__main__"}
    import matplotlib.pyplot as plt
    show = plt.show
    plt.show = lambda *a, **k: plt.close("all")
    try:
        for index, source in enumerate(notebook_cells()):
            try:
                exec(compile(source, f"<cell {index}>", "exec"), namespace)
            except Exception as exc:
                exc.cell_index = index
                raise
    finally:
        plt.show = show
        os.environ.clear()
        os.environ.update(saved)
    return namespace


def part2():
    cells = notebook_cells()
    check("notebook cell 3 is xai_cam.py verbatim",
          any(X.__doc__.strip()[:60] in c and "def deletion_insertion" in c for c in cells))
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        drive, run_dir, tar_path = build_fake_drive(root)
        ckpt_sha = hashlib.sha256((run_dir / "best.pt").read_bytes()).hexdigest()
        env = {
            "CS_DRIVE_PROJECT_ROOT": str(drive), "CS_LOCAL_ROOT": str(root / "content"),
            "CS_MODEL_NAME": "convnext_atto", "CS_INPUT_HW": f"{HW[0]},{HW[1]}",
            "CS_CHECKPOINT_SHA256": ckpt_sha, "CS_EXPECTED_TEST_ROWS": "30",
            "CS_PAIRS_PER_LABEL": "2", "CS_CURVE_STEPS": "8", "CS_DATASET_READ_MODE": "mount",
            "CS_DATASET_DOWNLOAD_CHUNK_MIB": "1", "CS_DATASET_DOWNLOAD_WORKERS": "3",
            "CS_PAPER_PAD_PX": str(PAD), "CS_TIMING_PAIRS": "3", "CS_TIMING_WARMUP": "1",
            "CS_CPU_TIMING_PAIRS": "2", "CS_CPU_SCORECAM_PROBE_CHANNELS": "4",
            "CS_BOOTSTRAP_ROUNDS": "200", "CS_BATCH_CANDIDATES": "16,8",
            "CS_LIBRARY_CHECK_PAIRS": "2", "CS_EXAMPLE_LABELS": "3", "CS_CPU_THREADS": "2",
        }
        out = drive / "new_runs" / "xai_cam_comparison_clean12_v1"

        # Run 1: full.
        try:
            ns = run_notebook(env)
            ok = True
        except Exception:
            traceback.print_exc()
            ok = False
        check("run 1: every cell runs", ok)
        if not ok:
            return
        pairs = pd.read_csv(out / "pairs.csv")
        metrics = pd.read_csv(out / "per_pair_metrics.csv")
        check("pairs: 2 per label, NORM excluded, true positives only",
              len(pairs) == 22 and "NORM" not in set(pairs["label"])
              and all(ns["predictions"].set_index("record_id").loc[r, f"pred_validation_tuned_{l}"] == 1
                      and ns["predictions"].set_index("record_id").loc[r, f"true_{l}"] == 1
                      for r, l in zip(pairs["record_id"], pairs["label"])))
        check("EMA weights were loaded (parity holds)",
              pd.read_csv(out / "parity.csv")["max_abs_diff_vs_reference"].max() < 1e-4)
        check("per_pair_metrics: one row per pair and method", len(metrics) == 22 * 4
              and set(metrics["method"]) == {"gradcam", "gradcampp", "scorecam", "random"})
        check("p_full matches the stored reference probability",
              np.allclose(metrics["p_full"], metrics["p_reference"], atol=1e-4))
        check("AUCs are in [0, 1]",
              metrics[["deletion_auc", "insertion_auc"]].stack().between(0, 1).all())
        summary = pd.read_csv(out / "summary.csv")
        check("summary has the four methods", list(summary["method"]) ==
              ["Grad-CAM", "Grad-CAM++", "Score-CAM", "Random map"])
        check("CIs bracket the means", ((summary["deletion_auc_ci_low"] <= summary["deletion_auc"])
                                         & (summary["deletion_auc"] <= summary["deletion_auc_ci_high"])).all())
        paired = pd.read_csv(out / "paired_vs_gradcam.csv")
        check("paired table: 3 comparisons x 3 metrics", len(paired) == 9)
        timing = pd.read_csv(out / "timing_summary.csv")
        check("timing summary has GPU-or-CPU and CPU rows for all three methods",
              set(timing["method"]) == {"Grad-CAM", "Grad-CAM++", "Score-CAM"}
              and timing["measured"].str.startswith("extrapolated").any())
        if (out / "library_check.csv").is_file():
            lib = pd.read_csv(out / "library_check.csv")
            check("library check r > 0.9999", lib["pearson_r"].min() > 0.9999, f"min r {lib['pearson_r'].min():.6f}")
        for name in ("summary.md", "figures/mean_curves.png", "figures/example_maps.png",
                     "eval_images.tar", "eval_images.tar.sha256", "config.json", "per_label.csv"):
            check(f"Drive has {name}", (out / name).is_file())
        n_images = pairs["record_id"].nunique()
        check("maps/ and curves/ hold one file per image",
              len(list((out / "maps").glob("*.npz"))) == n_images
              and len(list((out / "curves").glob("*.npz"))) == n_images)
        print((out / "summary.md").read_text())
        if os.environ.get("CS_VALIDATE_KEEP_DIR"):
            shutil.copytree(out / "figures", Path(os.environ["CS_VALIDATE_KEEP_DIR"]), dirs_exist_ok=True)

        # Run 2: empty runtime, archive gone -> Drive cache + resume.
        shutil.rmtree(root / "content")
        hidden = tar_path.with_suffix(".hidden")
        tar_path.rename(hidden)
        before = metrics.copy()
        try:
            ns2 = run_notebook(env)
            ok = True
        except Exception:
            traceback.print_exc()
            ok = False
        check("run 2: rerun on an empty runtime without the archive (Drive cache)", ok)
        if ok:
            after = pd.read_csv(out / "per_pair_metrics.csv")
            check("run 2: resume skipped every image (metrics unchanged)",
                  len(ns2["todo"]) == 0 and after.equals(before))
        hidden.rename(tar_path)

        # Run 3: a changed design under the same run name is refused.
        try:
            run_notebook({**env, "CS_CURVE_STEPS": "9"})
            refused = False
        except RuntimeError as exc:
            refused = "different design" in str(exc)
        check("run 3: changed design refused under the same CS_RUN_NAME", refused)

        # Run 4: wrong reference probabilities -> parity failure.
        preds = pd.read_csv(run_dir / "test_predictions.csv")
        preds.to_csv(run_dir / "test_predictions.good.csv", index=False)
        preds["prob_AF"] = (preds["prob_AF"] + 0.2).clip(0, 1)
        preds.to_csv(run_dir / "test_predictions.csv", index=False)
        try:
            run_notebook({**env, "CS_RUN_NAME": "parity_should_fail"})
            failed = False
        except RuntimeError as exc:
            failed = "differ from the reference run" in str(exc)
        check("run 4: wrong reference probabilities fail the parity check", failed)
        shutil.move(run_dir / "test_predictions.good.csv", run_dir / "test_predictions.csv")

        # Run 5: stage-only on a "CPU runtime", then the evaluation runtime uses its
        # cache with the big archive gone, through the compiled path if it compiles.
        staged = drive / "new_runs" / "staged"
        try:
            run_notebook({**env, "CS_RUN_NAME": "staged", "CS_STAGE_IMAGES_ONLY": "1",
                          "CS_LOCAL_ROOT": str(root / "content_stage")})
            stopped = False
        except SystemExit:
            stopped = True
        check("run 5: stage-only stops after caching the images",
              stopped and (staged / "eval_images.tar").is_file() and not (staged / "config.json").exists())
        tar_path.rename(hidden)
        try:
            ns5 = run_notebook({**env, "CS_RUN_NAME": "staged", "CS_COMPILE": "1",
                                "CS_LOCAL_ROOT": str(root / "content_eval")})
            ok = True
        except Exception:
            traceback.print_exc()
            ok = False
        hidden.rename(tar_path)
        check("run 5: evaluation runtime uses the staged cache (archive absent)", ok)
        if ok:
            bench = json.loads(next(staged.glob("benchmark_*.json")).read_text())
            check("run 5: compile attempted and a forward path chosen",
                  bench["forward_path"] in {"compiled", "eager"},
                  f"path={bench['forward_path']} compiled={bench['compiled_images_per_second']} "
                  f"eager={bench['eager_images_per_second']}")
            m5 = pd.read_csv(staged / "per_pair_metrics.csv")
            m1 = pd.read_csv(out / "per_pair_metrics.csv")
            merged = m5.merge(m1, on=["pair_id", "method"], suffixes=("_5", "_1"))
            check("run 5: AUCs equal run 1's (same design, either forward path)",
                  len(merged) == len(m1) and np.allclose(merged["deletion_auc_5"], merged["deletion_auc_1"], atol=1e-4)
                  and np.allclose(merged["insertion_auc_5"], merged["insertion_auc_1"], atol=1e-4))

        # cpu_timing.py on two of the sheets.
        import cpu_timing
        sheets = root / "sheets"
        sheets.mkdir()
        for i in range(2):
            sheet(i).save(sheets / f"s{i}.png")
        frame = cpu_timing.main(["--images", str(sheets), "--checkpoint", str(run_dir / "best.pt"),
                                 "--model-name", "convnext_atto", "--input-hw", f"{HW[0]},{HW[1]}",
                                 "--scorecam-probe", "4", "--threads", "2",
                                 "--out", str(root / "cpu.csv")])
        check("cpu_timing.py: 2 images x 3 methods", len(frame) == 6
              and set(frame["method"]) == {"gradcam", "gradcampp", "scorecam"})


if __name__ == "__main__":
    part1()
    part2()
    failed = [name for name, ok, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    sys.exit(1 if failed else 0)
