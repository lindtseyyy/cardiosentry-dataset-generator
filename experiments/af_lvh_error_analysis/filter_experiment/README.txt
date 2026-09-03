Filter-bandwidth regeneration experiment — 10 AF+LVH LVH-false-negative test cases
=====================================================================================

What this is
------------
The same 10 records were re-generated through the EXISTING synthetic-ECG pipeline
(image_pipeline stages 2-4, reused verbatim) with three band-pass settings:

    0.05-40 Hz    control (current release setting)
    0.05-100 Hz
    0.05-150 Hz

Everything else is byte-for-byte the same code path and parameters as the released
corpus: same staged record names, same kit render flags (150 DPI, 4-column layout,
full_mode II, calibration pulse, bw grid), same stage-4 augmentation RNG seeds
((record, 0), SEED 20260818), same JPEG settings.

VALIDATION: the regenerated 0.05-40 Hz control images are PIXEL-IDENTICAL to the
released prototype_1152 images (mean |diff| = 0.000 across all 10 cases, and the
grid_palette + rotation_deg columns match index.csv exactly). Therefore the only
thing that differs between the three bands below is the filter cut-off.

image_pipeline/config.py was NOT modified: BANDPASS is overridden in memory per
band and the file checksum was verified unchanged after the run.

Contents
--------
by_case/<case>/            one folder per ECG, for side-by-side comparison:
   0.05-40Hz.jpg           regenerated with the current filter (== released image)
   0.05-100Hz.jpg
   0.05-150Hz.jpg
   original_0.05-40Hz.jpg  the actual released image the model was evaluated on

by_band/<band>/            all 10 cases per band (ECG00xxxx_a0.jpg + kit .json)

staged/<band>/             the band-passed WFDB signals (.dat/.hea) behind each
                           image — for quantitative checks
rendered/<band>/           clean greyscale kit renders (pre-augmentation)

metadata.csv               one row per case x band: prototype labels, model LVH
                           probability, original annotations, grid palette,
                           rotation, and amplitude metrics (peak-based
                           Sokolow-Lyon proxy, max V5/V6, RMS)

Observations so far
-------------------
* Pixel change vs 40 Hz: 100 Hz affects 0.25-1.0% of pixels (|diff|>15 grey levels);
  150 Hz only slightly more than 100 Hz.
* Peak Sokolow-Lyon proxy gains +0.05...+0.22 mV at 100 Hz; the extra
  100 -> 150 Hz adds almost nothing (+0.002...+0.05 mV) — nearly all of the
  restored energy sits in the 40-100 Hz octave.
* The 40 Hz control reproduces the release exactly, so any inference differences
  on the regenerated images can be attributed to the filter alone.

Next step (not done yet): feed by_band/<band>/*.jpg back through the frozen
resnet50_hires model for controlled inference comparison.
