export interface ScoreBarProps {
  label: string;
  displayName: string;
  score: number;
  positive: boolean;
}

export default function ScoreBar({
  label,
  displayName,
  score,
  positive,
}: ScoreBarProps) {
  const clampedScore = Math.max(0, Math.min(1, score));

  return (
    <div className={`score-row ${positive ? "positive" : "negative"}`}>
      <div className="score-head">
        <span className="score-name">{displayName || label}</span>
        <span className="score-value" aria-label={`${label} model score`}>
          {clampedScore.toFixed(3)}
        </span>
      </div>
      <div className="score-bar-wrap">
        <div
          className="score-bar-fill"
          style={{ width: `${clampedScore * 100}%` }}
        />
      </div>
    </div>
  );
}
