import type { QualityMetrics } from "../api/types";

const FLAG_LABELS: Record<string, { label: string; poor?: boolean }> = {
  low_sharpness: { label: "Looks soft" },
  low_brightness: { label: "Too dark" },
  high_brightness: { label: "Too bright" },
  low_contrast: { label: "Washed out" },
  clipped_highlights: { label: "Highlights blown" },
  clipped_shadows: { label: "Shadows clipped" },
  glare: { label: "Possible glare" },
  small_sheet_area: { label: "Sheet small in frame" },
  extreme_perspective: { label: "Extreme perspective" },
  low_resolution: { label: "Low resolution" },
  grid_not_resolvable: { label: "Grid not resolvable in the model input" },
  poor_exposure: { label: "Exposure out of range", poor: true },
  poor_contrast: { label: "Contrast very low", poor: true },
  poor_clipping: { label: "Severe clipping", poor: true },
  poor_glare: { label: "Severe glare", poor: true },
  poor_area: { label: "Sheet very small in frame", poor: true },
  poor_perspective: { label: "Severe perspective", poor: true },
  poor_resolution: { label: "Very low resolution", poor: true },
  poor_sharpness: { label: "Very soft", poor: true },
};

const LEVEL_TITLES: Record<string, string> = {
  ok: "Image quality OK",
  warn: "Check image quality",
  poor: "Poor image quality",
};

export default function QualityBanner({
  quality,
}: {
  quality: QualityMetrics | null | undefined;
}) {
  if (!quality || !quality.flags || quality.flags.length === 0) return null;

  const title = LEVEL_TITLES[quality.level] ?? "Image quality";
  const unknownFlags = quality.flags.filter((f) => !FLAG_LABELS[f]);

  return (
    <div className={`quality-banner ${quality.level}`} role="status">
      <div className="q-title">{title}</div>
      <div className="q-flags">
        {quality.flags.map((flag) => {
          const def = FLAG_LABELS[flag];
          if (!def) return null;
          return (
            <span key={flag} className={`q-flag${def.poor ? " poor" : ""}`}>
              {def.label}
            </span>
          );
        })}
        {unknownFlags.map((flag) => (
          <span key={flag} className="q-flag">
            {flag}
          </span>
        ))}
      </div>
    </div>
  );
}
