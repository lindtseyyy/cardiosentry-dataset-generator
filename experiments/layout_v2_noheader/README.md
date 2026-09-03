# v2 renderer, no printed header — 5 validation images

Same v2 rendering path as `layout_v2_4col` (150 dpi, stock stroke widths,
`stage4_augment.py`, 4 leads per row). Two changes:

1. The **ID / Name / Date / Age / Sex block is gone**.
2. The top reserve shrank from 55 mm to 24 mm, so the rows spread up into the
   space the header used to occupy.

What is still on the sheet is only ECG: the grid, the twelve lead traces, the
lead names, the calibration pulses, the lead II rhythm strip, and the
`25 mm/s` / `10 mm/mV` footer.

## How it is switched

`config.PRINT_HEADER`, off by default, drives `--print_header` in
`KIT_RENDER_FLAGS`. `ECG_PRINT_HEADER=1` restores the v2 behaviour.

## Layout, before and after

| | header on | header off |
|---|---:|---:|
| header ink | 16.6 – 25.6 mm | **none** |
| lead row 1 (label / trace) | 42.2 / 55.2 mm | 11.2 / 24.2 mm |
| lead row 2 | 90.4 / 102.1 mm | 69.8 / 81.4 mm |
| lead row 3 | 137.8 / 149.9 mm | 126.0 / 139.9 mm |
| rhythm strip | 194.6 / 208.8 mm | 194.6 / 208.8 mm |
| row spacing | 47.3 mm | **56.4 mm** |
| top reserve | 5.5 units (55 mm) | 2.4 units (24 mm) |

## The reason this matters far more than cosmetics

The kit's `--print_header` does a bare dict lookup for `Age` and `Sex` and
raises `KeyError` if either is missing, so stage 1 has always **dropped any
record without both**. With the header off nothing reads those fields, and the
requirement disappears. Counted against the full v3 release:

| source | header ON | header OFF | gained |
|---|---:|---:|---:|
| CUHK | **153** | **17,951** | +17,798 |
| CHAPMAN | 7,639 | 7,716 | +77 |
| PTBXL | 10,141 | 10,141 | 0 |
| STEMI | 1,417 | 1,417 | 0 |
| **total** | **19,350** | **37,225** | **+17,875** |

CUHK records almost no age (43,145 of 67,432 source XMLs carry Philips'
`defaultage="50"` placeholder and 23,436 carry none at all — only 851 have a
real age), so the printed caption was excluding 98.5 % of the cohort.

By label combination:

| combination | header ON | header OFF | gained |
|---|---:|---:|---:|
| NORMAL | 12,771 | 12,839 | +68 |
| LVH | 2,320 | 9,699 | +7,379 |
| AF | 2,510 | 8,292 | +5,782 |
| STEMI | 1,480 | 3,402 | +1,922 |
| AF+LVH | 257 | 2,376 | +2,119 |
| STEMI+LVH | 7 | 401 | +394 |
| STEMI+AF | 5 | 173 | +168 |
| **STEMI+AF+LVH** | **0** | **43** | **+43** |

**The triple-positive class had zero images.** STEMI+LVH had 7 and STEMI+AF had
5. The STEMI co-morbidity that v3 was specifically built to supply — and that
v2 could not produce at all — was being silently destroyed at stage 1 by a
caption that carried no clinical information. This is the single most
consequential thing found in this pass, and it was invisible because stage 1
reported it only as an aggregate `skip_missing_metadata` count.

## Secondary benefit

The block carried nothing worth learning: the ID is our own synthetic
`ECGnnnnnn`, and `Name:` and `Date:` render blank on every sheet. Printed age
and sex are the only real information in it, and both are genuine confounders
for LVH and STEMI — a model that reads "Age: 82" off the paper is not reading
the ECG. Removing it takes that shortcut away.

## The 5 validation images

`validation/` — same 5 records as the previous two experiments, directly
comparable. 150 dpi, 1686x1311, ~590 KB, JPEG q85.

| file | record | source | label |
|---|---|---|---|
| ECG000001_a0.jpg | CUHK_002942 | CUHK | **STEMI+LVH** |
| ECG000002_a0.jpg | CUHK_000477 | CUHK | AF |
| ECG000003_a0.jpg | PTBXL_000002 | PTBXL | NORMAL |
| ECG000004_a0.jpg | CHAPMAN_000001 | CHAPMAN | AF |
| ECG000005_a0.jpg | STEMI_000004 | STEMI | STEMI |

## Reproducing

```bash
./run.sh
```

## Still open

- Stage 2 still writes `Age:` / `Sex:` header comments. Harmless — nothing reads
  them now — and left in place so `ECG_PRINT_HEADER=1` keeps working.
