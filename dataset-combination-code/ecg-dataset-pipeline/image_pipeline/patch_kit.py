"""Apply the local modifications ecg-image-kit needs, idempotently.

The kit is a separate checkout (see README), so a collaborator who clones it
fresh gets stock upstream code. Run this once after cloning; stage 3 checks it
has been applied and refuses to render otherwise.

Two changes are needed. Everything else the corpus wants is reachable through
the kit's own CLI flags.

  1. HEADER_PATCH    - inset the printed ID/Age/Sex block from the paper corner.
  2. LEADNAME_PATCH  - move the lead name above its trace instead of below it.
"""
from __future__ import annotations

import re
import sys

import config as C

MARKER = "PATCHED by image_pipeline/patch_kit.py"
MARKER_LEAD = "PATCHED-LEADNAME by image_pipeline/patch_kit.py"
# Distinct marker per patch site: a shared one makes the second patch believe
# the first one's marker was its own, and silently skip.
MARKER_RHYTHM = "PATCHED-RHYTHMNAME by image_pipeline/patch_kit.py"
MARKER_INK = "PATCHED-INKWIDTH by image_pipeline/patch_kit.py"
MARKER_COLS = "PATCHED-COLUMNS by image_pipeline/patch_kit.py"
MARKER_LAYOUT = "PATCHED-LAYOUT by image_pipeline/patch_kit.py"
MARKER_PAPER = "PATCHED-PAPER by image_pipeline/patch_kit.py"
MARKER_NP2 = "PATCHED-NUMPY2 by image_pipeline/patch_kit.py"
MARKER_LAZY = "PATCHED-LAZYDEPS by image_pipeline/patch_kit.py"
MARKER_HELPER = "PATCHED-UNUSED-IMGAUG by image_pipeline/patch_kit.py"
MARKER_FONT = "PATCHED-FIXEDFONT by image_pipeline/patch_kit.py"

HEADER_X = 1.6      # plot units right of the left edge
HEADER_Y = 1.1      # plot units down from the top edge

# The printed header sits flush against the top-left paper corner at
# x=0.05, y=y_max. Real printouts leave a visible margin, and a caption hard
# against the edge is the first thing lost to rotation or cropping.
#
# The whole block is replaced rather than just the initial assignment: the
# per-line loop resets x_offset to the literal 0.05 after every row, so
# patching only the top would indent the first line and leave the rest flush.
HEADER_PATCH = (
    """        x_offset = 0.05
        y_offset = int(y_max)
        printed_text, attributes, flag = generate_template(full_header_file)

        if flag:
            for l in range(0, len(printed_text), 1):

                for j in printed_text[l]:
                    curr_l = ''
                    if j in attributes.keys():
                        curr_l += str(attributes[j])
                    ax.text(x_offset, y_offset, curr_l, fontsize=lead_fontsize)
                    x_offset += 3

                y_offset -= 0.5
                x_offset = 0.05""",
    f"""        # {MARKER}: inset the printed header away from the paper corner.
        header_x = {{x}}
        x_offset = header_x
        y_offset = int(y_max) - {{y}}
        printed_text, attributes, flag = generate_template(full_header_file)

        if flag:
            for l in range(0, len(printed_text), 1):

                for j in printed_text[l]:
                    curr_l = ''
                    if j in attributes.keys():
                        curr_l += str(attributes[j])
                    ax.text(x_offset, y_offset, curr_l, fontsize=lead_fontsize)
                    x_offset += 3

                y_offset -= 0.5
                x_offset = header_x""",
)


# The kit draws each lead name BELOW the baseline:
#
#     t1 = ax.text(x_offset + x_gap + dc_offset,
#             y_offset-lead_name_offset - 0.2,
#             leadName, ...)
#
# Real clinical printouts put it above and to the left of the trace, clear of
# the waveform. Both call sites are patched: the 12 lead panels, and the
# full-mode rhythm strip at the bottom of the sheet.
LEADNAME_PATCH = (
    """            t1 = ax.text(x_offset + x_gap + dc_offset, 
                    y_offset-lead_name_offset - 0.2, 
                    leadName, 
                    fontsize=lead_fontsize)""",
    f"""            # {MARKER_LEAD}: label above the trace, not below it.
            t1 = ax.text(x_offset + x_gap, 
                    y_offset + {{dy}}, 
                    leadName, 
                    fontsize=lead_fontsize)""",
)

RHYTHM_PATCH = (
    """            t1 = ax.text(x_gap + dc_offset, 
                    row_height/2-lead_name_offset, 
                    full_mode, 
                    fontsize=lead_fontsize)""",
    f"""            # {MARKER_RHYTHM}: rhythm-strip label above the trace too.
            # NB the rhythm trace baseline is row_height/2 - lead_name_offset
            # + 0.8, not row_height/2; anchoring to the latter puts the label
            # BELOW the calibration pulse and the two collide.
            t1 = ax.text(x_gap, 
                    row_height/2 - lead_name_offset + 0.8 + {{dy}}, 
                    full_mode, 
                    fontsize=lead_fontsize)""",
)


# The kit hardcodes both stroke widths as matplotlib POINTS, which are
# resolution-independent, so raising DPI makes the sheet bigger without making
# the trace any finer. Measured against a real smartphone photo of a printout,
# the kit's 0.75 (x1.5 at the call site = 1.125 pt = 0.40 mm) is about twice
# the ~0.19 mm a thermal ECG head actually lays down.
#
# Patched to read the environment so a build can retune ink without editing the
# kit, and so an UNSET environment reproduces stock behaviour exactly - which is
# what keeps the already-published v2 corpus byte-reproducible.
INKWIDTH_PATCH = (
    """                   'grid_line_width' : 0.5,""",
    f"""                   # {MARKER_INK}: stroke widths from the environment.
                   'grid_line_width' : float(os.environ.get('ECG_GRID_LINE_WIDTH', 0.5)),""",
)

LINEWIDTH_PATCH = (
    """                   'line_width': 0.75,""",
    """                   'line_width': float(os.environ.get('ECG_LINE_WIDTH', 0.75)),""",
)


# ---------------------------------------------------------------- columns
# extract_leads.py hardcodes the per-column TIME OFFSET to `columns == 4`:
#
#     if columns == 4 and key in format_4_by_3[1]:
#         shiftedStart = start + int(rate*lead_length_in_seconds)
#     elif columns == 4 and key in format_4_by_3[2]:   ...
#
# With any other column count every branch is skipped, shiftedStart stays at
# `start`, and all N panels of a row plot the SAME opening segment of their
# lead. The sheet still looks plausible, which is what makes it dangerous: a
# 6-column render would silently show six copies of the first 1.67 s rather
# than a continuous 10 s. Generalised here to a lookup driven by config.yaml.
COLUMNS_PATCH = (
    """                shiftedStart = start
                if columns == 4 and key in format_4_by_3[1]:
                    shiftedStart = start + int(rate*lead_length_in_seconds)
                elif columns == 4 and key in format_4_by_3[2]:
                    shiftedStart = start + int(2*rate*lead_length_in_seconds)
                elif columns == 4 and key in format_4_by_3[3]:
                    shiftedStart = start + int(3*rate*lead_length_in_seconds)
                end = shiftedStart + int(rate*lead_length_in_seconds)""",
    "                # " + MARKER_COLS + ": column index -> time offset, any N.\n"
    "                _col = _column_index(key, columns, configs)\n"
    "                shiftedStart = start + int(_col*rate*lead_length_in_seconds)\n"
    "                end = shiftedStart + int(rate*lead_length_in_seconds)",
)

COLUMNS_GUARD_PATCH = (
    "                    if columns == 4 and key not in format_4_by_3[0]:",
    "                    if _col > 0:",
)

COLUMNS_HELPER = "\n".join([
    "",
    "",
    "def _column_index(lead, columns, configs):",
    "    # " + MARKER_COLS + ". Which column a lead sits in, and therefore its",
    "    # time offset. Reads format_<columns> from config.yaml, falling back to",
    "    # the original 4-column table so an unpatched config still behaves.",
    "    groups = configs.get('format_%d' % columns) or configs.get('format_4_by_3') or []",
    "    for i, g in enumerate(groups):",
    "        if lead in g:",
    "            return i",
    "    return 0",
    "",
])

# ---------------------------------------------------------------- layout
# The kit divides the sheet into `rows + 2` slots but only ever fills `rows` of
# them, and the two spare slots land at the TOP. Measured on a real render: the
# printed header ends 27 mm down and the first lead row does not start until
# 78 mm - a 51 mm dead band, a quarter of the sheet, while the traces are
# crowded into the lower two thirds.
#
# Replaced with an explicit layout: reserve a band at the top for the printed
# header and a margin at the bottom for the 25 mm/s footer, then spread the lead
# rows and the rhythm strip evenly over everything between them. y_base is the
# rhythm strip's baseline and the anchor the lead rows step up from, so it
# replaces every `row_height/2` the kit used for that purpose.
LAYOUT_PATCH = (
    "    #row_height = height * y_grid_size/(y_grid*(rows+2))\n"
    "    row_height = (height * y_grid_size/y_grid)/(rows+2)",
    "    # PATCHED-LAYOUT by image_pipeline/patch_kit.py: fill the sheet, and give the\n    # rhythm strip its OWN gap. The kit used one row_height for every gap, so\n    # tightening the twelve-lead rows also dragged the rhythm strip up with them.\n    _y_full = height * y_grid_size / y_grid\n    _bottom_reserve = float(os.environ.get('ECG_BOTTOM_RESERVE', 1.9))\n    _row_gap = float(os.environ.get('ECG_ROW_GAP', 4.4))\n    _rhythm_gap = float(os.environ.get('ECG_RHYTHM_GAP', 5.5))\n    y_base = _bottom_reserve\n    row_height = _row_gap\n    # The lead rows step up from the rhythm strip by _rhythm_gap, then by\n    # _row_gap each. Whatever is left over stays blank at the top of the sheet.\n    _first_gap = _rhythm_gap if full_mode != 'None' else _row_gap\n    _y_offset_init = y_base + _first_gap - row_height",
)



# The shiftedStart block appears TWICE in extract_leads.py at different
# indentation - once on the start_index != -1 path and once inside the
# start_index == -1 loop, which is the path stage 3 actually takes. Patching
# only the first (an exact-string replace does) leaves the live path hardcoded
# to 4 columns AND leaves a `_col` reference with nothing defining it, so a
# 6-column render dies with NameError. Matched by regex so both are caught.
COLUMNS_RX = re.compile(
    r"(?P<i>[ \t]*)shiftedStart = start\n"
    r"(?P=i)if columns == 4 and key in format_4_by_3\[1\]:\n"
    r".*?"
    r"(?P=i)end = shiftedStart \+ int\(rate\*lead_length_in_seconds\)", re.S)

GUARD_RX = re.compile(r"(?P<i>[ \t]*)if columns == 4 and key not in format_4_by_3\[0\]:")


def patch_columns(path) -> str:
    """Generalise the per-column time offset to any column count."""
    text = path.read_text()
    if MARKER_COLS in text:
        return "already patched"

    def repl(m):
        i = m.group("i")
        return (f"{i}# {MARKER_COLS}: column index -> time offset, any N.\n"
                f"{i}_col = _column_index(key, columns, configs)\n"
                f"{i}shiftedStart = start + int(_col*rate*lead_length_in_seconds)\n"
                f"{i}end = shiftedStart + int(rate*lead_length_in_seconds)")

    text, n = COLUMNS_RX.subn(repl, text)
    if n != 2:
        return f"FAILED: expected 2 shiftedStart blocks, matched {n}"
    text, g = GUARD_RX.subn(lambda m: m.group("i") + "if _col > 0:", text)
    if g != 2:
        return f"FAILED: expected 2 column guards, matched {g}"
    path.write_text(text + COLUMNS_HELPER)
    return f"patched ({n} blocks, {g} guards)"


PAPER_PATCH = (
    "        width = standard_values['width']\n"
    "        height = standard_values['height']",
    "        # " + MARKER_PAPER + ": paper size from the environment.\n"
    "        width = float(os.environ.get('ECG_PAPER_W_IN', standard_values['width']))\n"
    "        height = float(os.environ.get('ECG_PAPER_H_IN', standard_values['height']))",
)

XGAP_PATCH = (
    "    x_gap = np.floor(((x_max - (columns*secs))/2)/0.2)*0.2",
    "    # " + MARKER_PAPER + ": centre the row CONTENT (traces + calibration\n"
    "    # pulse). The kit centred the traces alone then quantised down to a\n"
    "    # 0.2 s step, leaving 9.8 mm left and 14.4 mm right.\n"
    "    x_gap = (x_max - (columns*secs + standard_values['dc_offset_length']))/2.0",
)


# ---------------------------------------------------------------- numpy 2
# imgaug 0.4.0 is unmaintained and reads `np.sctypes` at MODULE level
# (imgaug/imgaug.py:45-47), which numpy 2.0 removed. The kit imports imgaug
# unconditionally, so on any numpy>=2 environment - Colab, for one - every
# render dies with:
#
#     AttributeError: `np.sctypes` was removed in the NumPy 2.0 release.
#
# We never call imgaug (the kit's own augmentation layer is unused; stage 4
# does that work), so the import only has to SUCCEED. Restoring the three
# entries it reads is enough, and is prepended to the batch entry point so it
# runs before any import chain reaches imgaug.
#
# Not caught by local testing because this venv pins numpy 1.26, where the
# attribute still exists.
NUMPY2_SHIM = """import numpy as _np
if not hasattr(_np, "sctypes"):      # {marker}
    _np.sctypes = {{
        "int":     [_np.int8, _np.int16, _np.int32, _np.int64],
        "uint":    [_np.uint8, _np.uint16, _np.uint32, _np.uint64],
        "float":   [_np.float16, _np.float32, _np.float64],
        "complex": [_np.complex64, _np.complex128],
        "others":  [bool, object, bytes, str, _np.void],
    }}
"""


def patch_numpy2(path) -> str:
    """Prepend the numpy-2 compatibility shim to the kit's entry point."""
    text = path.read_text()
    if MARKER_NP2 in text:
        return "already patched"
    path.write_text(NUMPY2_SHIM.format(marker=MARKER_NP2) + text)
    return "patched"


# ---------------------------------------------------------------- lazy deps
# gen_ecg_image_from_data.py imports three helpers at module level:
#
#     from HandwrittenText.generate import get_handwritten   -> tensorflow, spacy,
#                                                               seaborn, cv2,
#                                                               validators, requests, bs4
#     from CreasesWrinkles.creases  import get_creased       -> skimage, imutils, cv2
#     from ImageAugmentation.augment import get_augment      -> imgaug, qrcode
#
# All three are called ONLY under `if(hw_text)`, `if(wrinkles)` and `if(augment)`,
# which default to False and which this pipeline never sets - stage 4 does that
# work instead. So the imports are pure cost: they drag ~1 GB of dependencies
# into every render subprocess, pay a ~7 s TensorFlow import on each of ~93 kit
# invocations, and turn any missing one into a hard failure of the whole build.
#
# Made lazy. Behaviour is identical if the flags are ever enabled - the import
# just happens at call time, and fails there with a clear error instead of at
# start-up.
LAZY_PATCH = (
    "from HandwrittenText.generate import get_handwritten\n"
    "from CreasesWrinkles.creases import get_creased\n"
    "from ImageAugmentation.augment import get_augment",
    "# " + MARKER_LAZY + ": import only if the (unused) flags are enabled.\n"
    "def get_handwritten(*a, **k):\n"
    "    from HandwrittenText.generate import get_handwritten as _f\n"
    "    return _f(*a, **k)\n"
    "def get_creased(*a, **k):\n"
    "    from CreasesWrinkles.creases import get_creased as _f\n"
    "    return _f(*a, **k)\n"
    "def get_augment(*a, **k):\n"
    "    from ImageAugmentation.augment import get_augment as _f\n"
    "    return _f(*a, **k)",
)


# helper_functions imports imgaug but never references it.  Keeping that dead
# import forces an abandoned NumPy-1-era package into every render process and
# is the only reason the FCMAE path needed imgaug at all.  Removing it lets the
# renderer use Colab's native, internally consistent NumPy/SciPy wheels.
UNUSED_IMGAUG_PATCH = (
    "from imgaug import augmenters as iaa",
    "# " + MARKER_HELPER + ": unused import removed",
)


# ---------------------------------------------------------------- fixed font
# Upstream chooses a random font for every record even when every augmentation
# flag is disabled. FCMAE pretraining needs exactly one image per ECG and the
# only permitted visual variation is grid colour, so let callers pin the font
# without changing the behaviour of existing builds that leave the variable
# unset.
FIXED_FONT_PATCH = (
    "        font = os.path.join('Fonts',random.choice(os.listdir(\"Fonts\")))",
    "        # " + MARKER_FONT + ": optional deterministic typeface.\n"
    "        _fixed_font = os.environ.get('ECG_FIXED_FONT', '').strip()\n"
    "        font = os.path.join('Fonts', _fixed_font) if _fixed_font else \\\n"
    "            os.path.join('Fonts', random.choice(os.listdir(\"Fonts\")))",
)


def patch_file(path, old: str, new: str, marker: str = MARKER) -> str:
    """Replace `old` with `new`, ignoring trailing whitespace differences.

    Upstream has trailing spaces on otherwise-blank lines, which makes an
    exact string match brittle across checkouts and editors.
    """
    text = path.read_text()
    if marker in text:
        return "already patched"

    pattern = re.compile(
        r"[ \t]*\n".join(re.escape(line.rstrip()) for line in old.splitlines())
    )
    match = pattern.search(text)
    if not match:
        return "FAILED: anchor text not found (kit version changed?)"
    path.write_text(text[:match.start()] + new + text[match.end():])
    return "patched"


def retune_offsets(path, dy) -> str:
    """Update the lead-name offset in an already-patched file.

    Without this, changing LEAD_NAME_DY in config.py would do nothing: the
    marker is already present, so patch_file() reports "already patched" and
    the old baked-in number stays. This rewrites just the number.
    """
    text = path.read_text()
    out, n = re.subn(r"(y_offset \+ )(-?\d+(?:\.\d+)?)", rf"\g<1>{dy}", text)
    out, n2 = re.subn(
        r"(row_height/2 - lead_name_offset \+ 0\.8 \+ )(-?\d+(?:\.\d+)?)",
        rf"\g<1>{dy}", out)
    if out != text:
        path.write_text(out)
        return f"retuned to {dy} ({n + n2} site(s))"
    return f"already at {dy}"


def is_patched() -> bool:
    target = C.KIT / "ecg_plot.py"
    if not target.exists():
        return False
    text = target.read_text()
    leads = C.KIT / "extract_leads.py"
    if not leads.exists() or MARKER_COLS not in leads.read_text():
        return False
    entry = C.KIT / "gen_ecg_images_from_data_batch.py"
    if not entry.exists() or MARKER_NP2 not in entry.read_text():
        return False
    helper = C.KIT / "helper_functions.py"
    if not helper.exists() or MARKER_HELPER not in helper.read_text():
        return False
    single = C.KIT / "gen_ecg_image_from_data.py"
    if (not single.exists() or MARKER_LAZY not in single.read_text()
            or MARKER_FONT not in single.read_text()):
        return False
    return all(m in text for m in (MARKER, MARKER_LEAD, MARKER_RHYTHM,
                                   MARKER_INK, MARKER_LAYOUT, MARKER_PAPER))


def main() -> int:
    target = C.KIT / "ecg_plot.py"
    if not target.exists():
        print(f"kit not found at {C.KIT}", file=sys.stderr)
        return 1

    failed = False

    old, new = HEADER_PATCH
    new = new.replace("{x}", str(HEADER_X)).replace("{y}", str(HEADER_Y))
    status = patch_file(target, old, new, MARKER)
    print(f"{target.name}: header        {status}")
    failed |= status.startswith("FAILED")

    dy = getattr(C, "LEAD_NAME_DY", 1.8)
    for label, (old, new), mk in (("lead names   ", LEADNAME_PATCH, MARKER_LEAD),
                                  ("rhythm strip ", RHYTHM_PATCH, MARKER_RHYTHM)):
        status = patch_file(target, old, new.replace("{dy}", str(dy)), mk)
        if status == "already patched":
            status = retune_offsets(target, dy)
        print(f"{target.name}: {label} {status}")
        failed |= status.startswith("FAILED")

    old, new = INKWIDTH_PATCH
    status = patch_file(target, old, new, MARKER_INK)
    print(f"{target.name}: ink width     {status}")
    failed |= status.startswith("FAILED")
    if status == "patched":                 # same marker guards both lines
        old, new = LINEWIDTH_PATCH
        s2 = patch_file(target, old, new, "os.environ.get('ECG_LINE_WIDTH'")
        print(f"{target.name}: trace width   {s2}")
        failed |= s2.startswith("FAILED")

    # --- layout: spread the rows over the whole sheet -------------------
    old, new = LAYOUT_PATCH
    status = patch_file(target, old, new, MARKER_LAYOUT)
    print(f"{target.name}: layout        {status}")
    failed |= status.startswith("FAILED")
    if status == "patched":
        text = target.read_text()
        # y_base replaces row_height/2 as the rhythm anchor, at every site.
        text = text.replace("    y_offset = (row_height/2)", "    y_offset = _y_offset_init")
        text = text.replace("row_height/2-lead_name_offset + 0.8",
                            "y_base-lead_name_offset + 0.8")
        text = text.replace("row_height/2 - lead_name_offset + 0.8",
                            "y_base - lead_name_offset + 0.8")
        target.write_text(text)
        print(f"{target.name}: rhythm anchor rebased on y_base")

    # --- paper size and horizontal centring -----------------------------
    old, new = PAPER_PATCH
    status = patch_file(target, old, new, MARKER_PAPER)
    print(f"{target.name}: paper size    {status}")
    failed |= status.startswith("FAILED")
    if status == "patched":
        old, new = XGAP_PATCH
        s2 = patch_file(target, old, new, "centre the row CONTENT")
        print(f"{target.name}: x centring    {s2}")
        failed |= s2.startswith("FAILED")

    # --- columns: generalise the per-column time offset -----------------
    leads = C.KIT / "extract_leads.py"
    if not leads.exists():
        print(f"extract_leads.py not found at {leads}", file=sys.stderr)
        return 1
    status = patch_columns(leads)
    print(f"{leads.name}: columns   {status}")
    failed |= status.startswith("FAILED")

    # --- numpy 2 compatibility for imgaug --------------------------------
    entry = C.KIT / "gen_ecg_images_from_data_batch.py"
    if entry.exists():
        print(f"{entry.name}: numpy2 shim  {patch_numpy2(entry)}")
    else:
        print(f"missing {entry}", file=sys.stderr); failed = True

    # --- remove helper_functions' dead imgaug import --------------------
    helper = C.KIT / "helper_functions.py"
    if helper.exists():
        old, new = UNUSED_IMGAUG_PATCH
        status = patch_file(helper, old, new, MARKER_HELPER)
        print(f"{helper.name}: unused imgaug {status}")
        failed |= status.startswith("FAILED")
    else:
        print(f"missing {helper}", file=sys.stderr); failed = True

    # --- make the unused heavy helpers lazy ------------------------------
    single = C.KIT / "gen_ecg_image_from_data.py"
    if single.exists():
        old, new = LAZY_PATCH
        status = patch_file(single, old, new, MARKER_LAZY)
        print(f"{single.name}: lazy deps  {status}")
        failed |= status.startswith("FAILED")

        old, new = FIXED_FONT_PATCH
        status = patch_file(single, old, new, MARKER_FONT)
        print(f"{single.name}: fixed font {status}")
        failed |= status.startswith("FAILED")
    else:
        print(f"missing {single}", file=sys.stderr); failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
