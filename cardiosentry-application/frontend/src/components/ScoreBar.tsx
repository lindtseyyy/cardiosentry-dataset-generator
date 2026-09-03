export interface ScoreBarProps {
  label: string;
  displayName: string;
  score: number;
  threshold: number;
  thresholdSource: string;
  positive: boolean;
  /** Blind mode: hide the score value and verdict until reference labels exist. */
  blind?: boolean;
  /** Effective threshold (may differ from `threshold` when an override slider is active). */
  effectiveThreshold?: number;
}

export default function ScoreBar({
  label,
  displayName,
  score,
  threshold,
  thresholdSource,
  positive,
  blind = false,
  effectiveThreshold,
}: ScoreBarProps) {
  const eff = typeof effectiveThreshold === "number" ? effectiveThreshold : threshold;
  const clampedScore = Math.max(0, Math.min(1, score));
  const clampedThr = Math.max(0, Math.min(1, eff));

  return (
    <div className={`score-row${blind ? " blind" : ""}`}>
      <div className="score-head">
        <span className="score-name">{displayName || label}</span>
        {blind ? (
          <span className="chip muted">hidden</span>
        ) : (
          <span className={`chip ${positive ? "pos" : "neg"}`}>
            {positive ? "Positive" : "Negative"}
          </span>
        )}
      </div>

      {blind ? (
        <div className="msg neutral" style={{ marginTop: 6 }}>
          scores hidden (blind mode) — enter reference labels to reveal
        </div>
      ) : (
        <>
          <div className="score-value" aria-label={`${label} model score`}>
            {clampedScore.toFixed(4)}
          </div>
          <div className="score-bar-wrap">
            <div
              className="score-bar-fill"
              style={{ width: `${clampedScore * 100}%` }}
            />
            <div
              className="score-bar-tick"
              style={{ left: `calc(${clampedThr * 100}% - 1.5px)` }}
            />
          </div>
          <div className="score-meta">
            <span>model score {clampedScore.toFixed(4)}</span>
            <span>
              threshold {clampedThr.toFixed(4)} ({thresholdSource})
            </span>
          </div>
        </>
      )}
    </div>
  );
}
