"""Rule out LBBB/conduction disease as the reason 55827005 shows high voltage.
LBBB widens QRS to >120 ms; LVH leaves it near-normal."""
import ast, csv, glob, os, random, re
import numpy as np, wfdb
from concurrent.futures import ProcessPoolExecutor

CH="/home/lindtseyxvii/Codes/cardiosentry/dataset-files/chapman-dataset"
PT="/home/lindtseyxvii/Codes/cardiosentry/dataset-files/ptb-xl-dataset"
rng=random.Random(42)

def qrs_ms(path):
    try:
        s=np.asarray(wfdb.rdrecord(path).p_signal,float)
        if s.shape!=(5000,12) or not np.isfinite(s).all(): return None
        s=s-np.median(s,axis=0)
        vm=np.sqrt((np.diff(s,axis=0)**2).sum(axis=1))        # spatial velocity
        vm=np.convolve(vm,np.ones(15)/15,mode="same")
        thr=np.percentile(vm,99)*0.5
        pk=[]; i=0
        while i<len(vm):                                       # crude R detection
            if vm[i]>thr:
                j=i
                while j<len(vm) and vm[j]>thr: j+=1
                pk.append((i+j)//2); i=j+100
            else: i+=1
        if len(pk)<4: return None
        w=[]
        for p in pk[1:-1]:
            a,b=max(0,p-100),min(len(vm),p+100)
            seg=vm[a:b]; t=seg.max()*0.20
            idx=np.where(seg>t)[0]
            if len(idx): w.append((idx[-1]-idx[0])*1000/500)
        return float(np.median(w)) if w else None
    except Exception: return None

def run(tag,paths):
    with ProcessPoolExecutor(max_workers=3) as ex:
        r=[x for x in ex.map(qrs_ms,paths,chunksize=16) if x]
    r=np.array(r)
    print(f"  {tag:34s} n={len(r):4d}  QRS median={np.median(r):6.1f} ms   >120ms={100*np.mean(r>120):5.1f}%")

recs={}
for h in glob.glob(f"{CH}/WFDBRecords/*/*/*.hea"):
    t=open(h).read(); m=re.search(r"#Dx:\s*(.*)",t)
    recs[h[:-4]]=set(c.strip() for c in m.group(1).split(",") if c.strip()) if m else set()
A,B,SR="164873001","55827005","426783006"
gA=[p for p,c in recs.items() if A in c and B not in c]
gB=[p for p,c in recs.items() if B in c and A not in c]
gC=[p for p,c in recs.items() if not (c-{SR})]
for g in (gB,gC): rng.shuffle(g)

rows=list(csv.DictReader(open(f"{PT}/ptbxl_database.csv")))
def pt(sel,n):
    o=[f"{PT}/{r['filename_hr']}" for r in rows
       if sel(ast.literal_eval(r["scp_codes"])) and os.path.exists(f"{PT}/{r['filename_hr']}.dat")]
    rng.shuffle(o); return o[:n]

print("QRS DURATION (LBBB discriminator)")
run("PTB-XL NORM (narrow control)",   pt(lambda d:"NORM" in d and len(d)<=2,500))
run("PTB-XL CLBBB (WIDE control)",    pt(lambda d:"CLBBB" in d,500))
run("PTB-XL LVH",                     pt(lambda d:"LVH" in d,500))
run("Chapman 164873001 (doc. LVH)",   gA[:500])
run("Chapman 55827005 (undefined)",   gB[:500])
run("Chapman SR-only controls",       gC[:500])
