# CUHK Philips XML corpus — compatibility assessment

Source: [Zenodo 5700069](https://zenodo.org/records/5700069), "Heart Disease XML
Strain", Faculty of Medicine, The Chinese University of Hong Kong. CC0.
Local copy: `dataset-files/cuhk_dataset/` — 16,755 XML files, 3.1 GB.

**Verdict: format-compatible and large, but blocked on a decoder, and the labels
are a grade weaker than anything currently in the corpus.** Details below.

---

## 1. Format — compatible, with one blocker

Philips PageWriter XL carts exported via TraceMasterVue. UTF-16 XML,
`PhilipsECG.xsd` v1.04.

| Property | CUHK | Pipeline needs | |
|---|---|---|---|
| Sampling rate | 500 Hz | 500 Hz (`C.FS`) | ✅ native, no resampling |
| Leads | 12 | 12 (`C.LEAD_ORDER`) | ✅ |
| Lead naming | `C1..C6` or `V1..V6` | `V1..V6` | ✅ trivial fold |
| Duration | 11 s | 10 s (5000 samples) | ✅ trim |
| Resolution | 5 µV/LSB | — | ✅ |
| Age | **absent in 100%** | required by the kit | ⚠️ see §4 |
| Sex | present in 77.4% | required by the kit | ⚠️ |
| **Waveform access** | **XLI-compressed** | WFDB-readable | ❌ **blocker** |

`stage2_transcode.py` would accept these unchanged once the samples are in hand
— the specs line up better than any source already in the project.

### The XLI blocker

The 11 s rhythm is compressed with Philips XLI, which is proprietary. I got part
of the way; `philips_xml.py` records exactly where it stops.

- **Solved — container.** The payload is a chain of 16 chunks, each
  `int32 size, int16, int16` + `size` bytes. Consumes the payload exactly, zero
  remainder, on every file tested.
- **Solved — entropy layer.** Each chunk is LZW with **fixed 10-bit codes,
  MSB-first, dictionary capped at 1024 entries then frozen** (no width growth,
  no reset). Every wrong variant desynchronises within a few hundred codes;
  this one runs clean to the end of every chunk. Each decodes to ~11,000 bytes
  = 5,500 int16 = exactly 11 s at 500 Hz. See `lzw_decompress`.
- **Not solved — residual layer.** Those int16 are residuals under a predictor
  I did not identify. Plane-split and interleaved byte orders × first- and
  second-order integration were all scored against the cart's own per-lead R/S
  amplitudes: best correlation +0.43, and chunks 6–15 decode to *identical*
  content, which cannot be true of V1/V2/V3. Also unexplained: 16 chunks for 12
  leads.

Finishing it needs the reference algorithm, not more guessing. The public
reverse-engineering of Philips XLI (Watford's `XliDecompressor`) covers the
residual step; with the LZW layer already done, that is a bounded port.

### What works today without XLI

**10,019 of 16,755 files** also carry the 12-lead representative beat as plain
base64 int16 — no compression. It decodes correctly: per-lead peak-to-peak
correlates **+0.986** with the R/S/Q amplitudes the cart wrote into the same
file, across all 12 leads. That is not something a wrong decode produces.

But it is **600 samples = 1.2 s per lead**. The image pipeline renders 10 s
(5,000 samples) per sheet. A representative beat cannot fill that, and tiling
one to 10 s would fabricate rhythm the patient never had — which the project's
own rules forbid, and which would wreck the AF head in particular.

So the median beats are useful for signal-level work (ST measurement, a
beat-based model) but **cannot feed the existing renderer**.

---

## 2. Labels — usable, but a grade weaker

### The folder names are not diagnoses

`STEMI XML AB` / `CE` / `GZ` are **patient-ID batches**, not severity or
diagnosis groups. Severity distribution is near-identical across all three
(≈87% ABNORMAL each), and the patient-ID prefixes partition cleanly by folder.

### It is a longitudinal registry, not a case series

**16,755 ECGs from 1,341 patients, recorded 2005–2019.** (1,341 is the count
from the `<patientid>` field; an earlier estimate of ~1,600 came from parsing
filenames, which over-splits.) Individual patients
contribute up to 13+ tracings across 14 years. Labelling every ECG in
`STEMI XML AB` as STEMI would mean calling a 2017 follow-up tracing a STEMI
because the patient infarcted in 2006 — a worse error than the confound this
whole exercise exists to fix.

Any split must group on patient. `inventory.py` reports patient counts beside
every ECG count for exactly this reason.

### Per-ECG labels do exist

Each file carries Philips criteria **coded statements** — the same structural
role as PTB-XL's SCP codes. The vocabulary is unusually well suited here,
because it names the benign ST-elevation mimics with their own codes instead of
lumping them into injury:

| Group | Example codes | Example text |
|---|---|---|
| Acute infarct | `IMIA`, `AMIA`, `EAMIA` | "INFARCT, ACUTE — Q>35mS, ST>0.10mV, II III aVF" |
| ST elevation / injury | `BSTEI`, `CINJA`, `PINJI` | "ST ELEVATION, CONSIDER ANTERIOR INJURY" |
| **Benign mimic** | `EREPOL`, `STE95`, `STELVH` | "ST ELEV, PROBABLE NORMAL EARLY REPOL PATTERN" |

### 🔴 Every statement is machine-generated

All 16,755 files carry `source="Analysis program"` on every coded statement.
**There is not one physician over-read in the corpus**, and `status="New"`
throughout means unconfirmed.

That places these labels a clear grade below what the project already has:

| Source | Label provenance | Grade |
|---|---|---|
| LUDB | cardiologist wrote `STEMI: <wall>` | gold, asserted |
| PTB-XL curated | human inspected each trace | proxy, human |
| Chongqing (training) | dedicated STEMI column | asserted |
| **CUHK** | **cart algorithm, unconfirmed** | **machine** |

The Philips acute-MI statements have well-known modest positive predictive
value. Used as a test-set gold standard they would be misleading; used as a
*training* signal or a large secondary cohort, they are reasonable.

---

## 3. Volume — this is the part that is genuinely large

Full-corpus scan (`inventory.py`). "beats-ok" = decodable today, without XLI.

| Group | ECGs | Patients | beats-ok ECGs | beats-ok patients |
|---|---:|---:|---:|---:|
| Acute infarct asserted | 1,657 | 491 | 1,180 | ~330 |
| ST elevation / injury (not acute) | 1,229 | ~480 | 442 | ~220 |
| **Either acute or ST elevation** | **2,884** | **772** | 1,622 | ~460 |
| Benign ST-elevation mimic | 510 | ~230 | 356 | ~160 |
| Atrial fibrillation | 692 | 227 | — | — |
| Atrial flutter (excluded from AF) | 48 | 37 | — | — |
| LVH | 1,264 | 356 | — | — |
| Severity NORMAL | 690 | 301 | — | — |
| Severity DEFECTIVE (drop) | 135 | 96 | — | — |
| **Total** | **16,755** | **1,341** | 10,019 | ~1,150 |

### Against what we already have

| Cohort | STEMI+ | Patients | Label grade |
|---|---:|---:|---|
| LUDB (gold) | 8 | 8 | cardiologist |
| PTB-XL curated (proxy) | 58 | 58 | human-inspected |
| **CUHK acute infarct** | **1,657** | **491** | machine |
| CUHK acute infarct, sex present | 1,265 | **388** | machine |
| Chongqing (training) | 1,442 | — | asserted |

**CUHK is the only candidate at the scale of the training corpus itself.** 491
patients with an asserted acute-infarct statement is 61× the gold cohort, and
comparable in size to all 1,442 Chongqing STEMI records. Requiring the sex the
renderer needs trims it to 1,265 ECGs from 388 patients.

The 510 benign-mimic ECGs are separately valuable — that is the hardest
negative class there is, and the external set currently has only 8 of them.

---

## 4. Other issues found

- **Age is missing in 100% of the corpus.** Not 60% — an earlier reading
  counted the `AGEUNK` *criteria statement*, which only means the cart had no
  age at analysis time. The authoritative field is `<age/>` in the patient
  block, and it is **empty in all 16,755 files**; the corpus is age-stripped as
  part of anonymisation. Sex survives in 77.4% (Male 10,023 / Female 2,947 /
  Unknown 3,785). The kit needs both, so age must be synthesised — see
  "Substituting age" below.
- **135 ECGs are flagged DEFECTIVE** by the cart. Drop them outright.
- **A second source confound is available for free.** Chongqing STEMI came from
  one vendor; CUHK is Philips. If both end up in training, vendor and label are
  no longer collinear — which is the actual fix for the problem the external
  test set only *measures*.

---

## 4b. 🔴 Substituting age — and a leak this question uncovered

Age has to be synthesised for CUHK (100% missing) or the kit's `--print_header`
raises and `stage1` drops every record. The obvious fix — a constant, or the
cohort mean — is the one thing that must not be done, and checking why turned up
a defect in the corpus the models were already trained on.

### The model can read the header

The kit prints `ID:` / `Age:` / `Sex:` on every sheet. Cropping that region from
a real training sheet and downscaling exactly as `data.py` does (1686x1311 ->
384x384 bicubic) leaves the text **plainly legible** — roughly 9 px glyph
height, and `ID: ECG006571 / Age: 70 yrs / Sex: Female` reads without effort.
It is not lost to resizing; it is in the tensor.

### Consequence 1 — a constant age would be a source tag

Age in the balanced corpus is nearly identical across sources (CHAPMAN 64.9,
PTB-XL 62.0, Chongqing 64.5), so today the printed age carries almost no source
information. Stamp every CUHK sheet with one value and it becomes a perfect
one: any sheet reading that exact age is CUHK, and CUHK would be the STEMI-rich
arm. That rebuilds the confound this whole effort exists to remove.

Age does correlate with LABEL — NORMAL 51.7 vs AF 71.7, LVH 67.6, STEMI 64.5 —
so a readable age is a genuine shortcut for "normal vs pathological" even
without a source tag.

**Therefore:** sample CUHK ages from the existing corpus's own age distribution
(mean ~64, sd ~16), conditioned on nothing. That leaves the printed age
uninformative about source, which is the property that matters. Record the
synthesis in the metadata so no one later mistakes it for real.

### Consequence 2 — the printed ID already leaks the label, today

Checking whether age leaks exposed that the record ID does, and far worse. In
`prototype_1152`, stage 2's sequential `ECG%06d` numbering ran in source order:

| Source | printed ID range |
|---|---|
| CHAPMAN | ECG000002 - ECG003241 |
| PTB-XL | ECG003242 - ECG007796 |
| Chongqing (all STEMI) | ECG007797 - ECG009238 |

`ID >= ECG007797` predicts `STEMI=1` with **accuracy 1.000** — 1,442 of 1,442
above the threshold are STEMI, 0 of 5,228 below it are. That threshold is
printed, legibly, on every sheet the models trained on.

So the STEMI head had a fourth way to reach 0.9861 AUROC, alongside ST
morphology and scanner appearance: **read a four-digit number**. This is not
hypothetical - the legibility is demonstrated above.

The external test set is unaffected: `01_build_metadata.py` shuffles before
staging precisely so the printed ID carries no source or label information, and
the leakage probe confirmed it (0.562 -> 0.491 source accuracy, chance 0.519).
That means the pending Colab run discriminates this hypothesis too, at no extra
cost — a model that learnt the number collapses on the external set exactly as
one that learnt the scanner does.

**For any retraining, shuffle before staging.** It is a one-line change in the
input CSV ordering and it removes a perfect answer key from the sheets.

---

## 5. Recommendation

**For the external test set: do not use it as gold.** Machine labels cannot
adjudicate whether a model learnt ST elevation — they are themselves an
algorithm's opinion of ST elevation, so agreement partly measures agreement with
Philips' criteria rather than with the truth. Keep LUDB as gold. CUHK could join
as a clearly-labelled third tier, reported separately, where its scale buys
tight confidence intervals that 8 gold positives never will.

**For training: this is the highest-value use, and it addresses the root cause.**
Right now 100% of STEMI positives come from one database, which is *why* the
head is unauditable. Adding ~1,657 acute-infarct ECGs from ~522 Hong Kong
patients on Philips hardware would make STEMI multi-source — after which the
per-source probe works on the STEMI head like it already does on AF and LVH, and
the confound stops being structural.

**The gate on both is the XLI decoder.** Median beats cannot drive the renderer,
so no XLI means no sheets means neither use case is reachable. That is the one
task worth doing next, and `lzw_decompress` means it is the last stage rather
than the whole problem.

Sequence I would suggest:

1. Port the XLI residual step; validate against `<ramp>`/`<samp>` per lead —
   `philips_xml.py` already reads both, so the check is one function call.
2. Re-run `inventory.py`; the "beats-ok" columns become the full columns.
3. Build a patient-grouped CUHK cohort, one ECG per patient, using
   `ACUTE_INFARCT` only, excluding DEFECTIVE and every mimic.
4. Decide training-vs-test *after* seeing the current external results — if the
   models already fail on LUDB, CUHK's value is as a training fix, not another
   measurement.

## Files

```
philips_xml.py   reader: statements, measurements, median beats, XLI LZW layer
inventory.py     corpus scan; ECG and PATIENT counts, decodability
FINDINGS.md      this document
```
