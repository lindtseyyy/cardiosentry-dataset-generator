# Final render settings — 5 validation images

The settings the full corpus will be generated with. Nothing here is
experimental; every value was validated in an earlier experiment and this
build only confirms that the accumulated edits did not disturb each other.

## Settings

| | value |
|---|---|
| renderer | `stage4_augment.py` (the v2 path) |
| resolution | 150 dpi, 1686 x 1311 px, JPEG q85 4:2:0 |
| signal band | 0.05 – 150 Hz, zero-phase — now the **config default** |
| layout | 3 x 4 + lead II rhythm strip, 25 mm/s, 10 mm/mV |
| printed header | **off** — no ID / Name / Date / Age / Sex |
| lead-row gap | 4.4 units = 44 mm |
| rhythm-strip gap | 5.5 units = 68 mm |
| horizontal margins | 12.0 mm left / 12.4 mm right (content-centred) |

`stage4_photo.py` and `photo_config.py` remain on disk but are not used.

## Verified against the approved build

Trace baselines and gaps are identical to `layout_v2_tight`, the build that was
signed off:

```
approved  traces [53.8  97.9 140.9 208.8] mm   gaps [44.1 43.0 67.9] mm
FINAL     traces [53.8  97.9 140.9 208.8] mm   gaps [44.1 43.0 67.9] mm
```

The one deliberate difference is horizontal. The kit centred the traces alone
and then quantised the left gap down to a 0.2 s step, so the calibration pulse
pushed everything right: 9.8 mm left, 14.4 mm right. Centring on the real
content width (traces + pulse) gives 12.0 / 12.4 mm. Measured on the reference
smartphone photo the real printout's left margin is 15.3 mm, so the centred
version is the closer fit — though the reference's right edge runs out of frame
entirely, which says more about how these get photographed than about margins.

Content width is unchanged either way: 255 mm on a 279 mm sheet, 91.3 % fill.
The trace itself is fixed at 25 mm/s x 10 s = 250 mm and cannot be widened.

## Removed

- `ECG_NUM_COLUMNS` and all 6-column support (kit `format_6` tables, the per-N
  lead-order lookup, the `layout_v2_6col` reproducer). Columns are hardcoded
  to 4, the clinical standard.
- The `extract_leads.py` fix is **kept**: the kit hardcoded the per-column time
  offset to `columns == 4`, and with any other value it silently replays the
  same opening segment in every panel. Behaviour at 4 columns is unchanged;
  it is retained as a guard.

## The 5 images

| file | record | source | label |
|---|---|---|---|
| ECG000001_a0.jpg | CUHK_002942 | CUHK | **STEMI+LVH** |
| ECG000002_a0.jpg | CUHK_000477 | CUHK | AF |
| ECG000003_a0.jpg | PTBXL_000002 | PTBXL | NORMAL |
| ECG000004_a0.jpg | CHAPMAN_000001 | CHAPMAN | AF |
| ECG000005_a0.jpg | STEMI_000004 | STEMI | STEMI |

## Storage estimate for the full corpus

37,225 records survive stage 1. At `VARIANTS_PER_SPLIT = {train: 4, val: 1,
test: 1}` and an 80/10/10 grouped split (train 30,717 / val 3,286 / test 3,222):

**129,376 images.**

| artefact | size | note |
|---|---:|---|
| final JPEGs | **72.8 GB** | the corpus |
| annotation JSON sidecars | 45.0 GB | lead boxes + traced pixel paths, one per image |
| `rendered/` clean PNGs | 9.1 GB | intermediate, deletable after stage 4 |
| `staged/` WFDB | 4.2 GB | intermediate, deletable after stage 3 |
| **peak during the build** | **131.1 GB** | all four coexist |
| **after cleanup** | **117.8 GB** | JPEG + JSON |
| JPEG only | 72.8 GB | if the sidecars are not needed |

Measured from this build: 590 KB per JPEG, 365 KB per JSON, 255 KB per
rendered PNG, 118 KB per staged record.

Scaling by train variants:

| train variants | images | JPEG | with JSON |
|---:|---:|---:|---:|
| 4 (current) | 129,376 | 72.8 GB | 117.8 GB |
| 3 | 98,659 | 55.5 GB | 89.8 GB |
| 2 | 67,942 | 38.2 GB | 61.9 GB |
| 1 | 37,225 | 21.0 GB | 33.9 GB |

The sidecars are 38 % of the finished corpus. They carry the kit's lead
bounding boxes and traced pixel paths — useful for segmentation or digitisation
work, dead weight for plain classification. Dropping them is the cheapest single
saving available.

## Filter safety

The band used to default to 40 Hz (monitoring) with 150 Hz supplied through
`ECG_LOWPASS_HZ` per build. That was the wrong way round: a run launched in a
shell that had not exported the variable would silently produce a 40 Hz corpus,
and the sheets look **identical** at either cutoff — the only way to tell is to
re-filter the source and diff. Three changes close that:

1. `config.BANDPASS` now defaults to **0.05–150 Hz**, so the safe value is what
   you get by accident. `ECG_LOWPASS_HZ` still overrides for filter variants.
2. Stage 2 prints the band and where it came from, and warns loudly below
   100 Hz.
3. `staged_map.csv` carries a `bandpass_hz` column per row, and the `staged/`
   skip-stamp now fingerprints **(manifest, band)** rather than the manifest
   alone — previously a re-run with a different band found a matching stamp,
   skipped stage 2, and rendered the old band's signals under the new build.

Verified on the final build by reconstruction:

```
final build vs 0.05-40  Hz : max dev  186.716 uV
final build vs 0.05-150 Hz : max dev    0.500 uV   <- one int16 LSB, i.e. quantisation
```

## Reproducing

```bash
./run.sh
```
