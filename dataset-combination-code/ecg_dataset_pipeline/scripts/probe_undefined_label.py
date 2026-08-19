"""Signal-level test: does 55827005 behave like LVH on the actual ECG?

LVH has objective, sex-adjusted voltage criteria. If 55827005 means LVH,
records carrying it should show elevated Sokolow-Lyon / Cornell voltages
comparable to the documented LVH code. If it means something else, they
should look like controls.
"""
import ast, csv, glob, io, os, random, re, sys
import numpy as np, wfdb
from concurrent.futures import ProcessPoolExecutor

CH = "/home/lindtseyxvii/Codes/cardiosentry/dataset-files/chapman-dataset"
PT = "/home/lindtseyxvii/Codes/cardiosentry/dataset-files/ptb-xl-dataset"
L = {n: i for i, n in enumerate(["I","II","III","AVR","AVL","AVF","V1","V2","V3","V4","V5","V6"])}
rng = random.Random(42)

def metrics(path):
    try:
        sig = np.asarray(wfdb.rdrecord(path).p_signal, float)
        if sig.shape != (5000, 12) or not np.isfinite(sig).all():
            return None
        sig = sig - np.median(sig, axis=0)              # baseline
        hi = np.percentile(sig, 99.5, axis=0)           # tallest R
        lo = np.percentile(sig, 0.5, axis=0)            # deepest S
        sl = abs(lo[L["V1"]]) + max(hi[L["V5"]], hi[L["V6"]])
        cor = hi[L["AVL"]] + abs(lo[L["V3"]])
        return sl, cor, hi[L["V5"]], abs(lo[L["V1"]])
    except Exception:
        return None

def run(tag, items):
    """items: list of (path, sex) where sex in {'M','F','?'}"""
    with ProcessPoolExecutor(max_workers=3) as ex:
        res = list(ex.map(metrics, [p for p, _ in items], chunksize=16))
    keep = [(r, s) for r, (_, s) in zip(res, items) if r]
    sl = np.array([r[0] for r, _ in keep]); cor = np.array([r[1] for r, _ in keep])
    sex = [s for _, s in keep]
    sl_pos = 100 * np.mean(sl >= 3.5)
    thr = np.array([2.0 if s == "F" else 2.8 for s in sex])
    cor_pos = 100 * np.mean(cor >= thr)
    either = 100 * np.mean((sl >= 3.5) | (cor >= thr))
    print(f"  {tag:34s} n={len(keep):4d}  SL median={np.median(sl):5.2f} mV  "
          f"SL+={sl_pos:5.1f}%  Cornell+={cor_pos:5.1f}%  either+={either:5.1f}%")
    return sl

# ---------- Chapman groups ----------
recs = {}
for h in glob.glob(f"{CH}/WFDBRecords/*/*/*.hea"):
    t = open(h).read(); m = re.search(r"#Dx:\s*(.*)", t); s = re.search(r"#Sex:\s*(.*)", t)
    codes = set(c.strip() for c in m.group(1).split(",") if c.strip()) if m else set()
    sx = "F" if s and s.group(1).strip().lower().startswith("f") else "M"
    recs[h[:-4]] = (codes, sx)

A, B, SR = "164873001", "55827005", "426783006"
gA  = [(p, s) for p, (c, s) in recs.items() if A in c and B not in c]
gB  = [(p, s) for p, (c, s) in recs.items() if B in c and A not in c]
gC  = [(p, s) for p, (c, s) in recs.items() if not (c - {SR})]
# age/comorbidity-matched-ish control: has flutter but neither LVH code
gD  = [(p, s) for p, (c, s) in recs.items() if "164890007" in c and A not in c and B not in c]
for g in (gB, gC, gD): rng.shuffle(g)

print("CHAPMAN / NINGBO")
run("164873001 (documented LVH)", gA)
run("55827005 (undefined)",        gB[:900])
run("SR-only controls",            gC[:900])
run("flutter, neither LVH code",   gD[:900])

# ---------- PTB-XL reference ----------
rows = list(csv.DictReader(open(f"{PT}/ptbxl_database.csv")))
def pt(sel, n):
    out = []
    for r in rows:
        d = ast.literal_eval(r["scp_codes"])
        if sel(d):
            p = f"{PT}/{r['filename_hr']}"
            if os.path.exists(p + ".dat"):
                out.append((p, "F" if str(r["sex"]) == "1" else "M"))
    rng.shuffle(out); return out[:n]

print("\nPTB-XL (independent reference standard)")
run("LVH (diagnostic statement)",  pt(lambda d: "LVH" in d, 900))
run("VCLVH (voltage criteria only)", pt(lambda d: "VCLVH" in d and "LVH" not in d, 900))
run("NORM (normal ECG)",           pt(lambda d: "NORM" in d and len(d) <= 2, 900))
