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
    return all(m in text for m in (MARKER, MARKER_LEAD, MARKER_RHYTHM))


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

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
