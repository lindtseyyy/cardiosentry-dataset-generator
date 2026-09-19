import ScoreBar from "./ScoreBar";

export interface PredictionLabelsProps {
  labels: string[];
  displayNames: Record<string, string>;
  scores: Record<string, number>;
  positive: string[];
  blind?: boolean;
}

export default function PredictionLabels({
  labels,
  displayNames,
  scores,
  positive,
  blind = false,
}: PredictionLabelsProps) {
  if (blind) {
    return <div className="panel muted center">Results hidden until labels are entered.</div>;
  }

  const positiveSet = new Set(positive);
  const positiveLabels = labels.filter((label) => positiveSet.has(label));
  const negativeLabels = labels.filter((label) => !positiveSet.has(label));

  return (
    <div className="prediction-labels">
      <div className="label-section-head">
        <h2>Positive labels</h2>
        <span className="count-badge">{positiveLabels.length}</span>
      </div>

      {positiveLabels.length ? (
        positiveLabels.map((label) => (
          <ScoreBar
            key={label}
            label={label}
            displayName={displayNames[label] ?? label}
            score={scores[label]}
            positive
          />
        ))
      ) : (
        <div className="panel muted center">No positive labels</div>
      )}

      {negativeLabels.length ? (
        <details className="negative-labels">
          <summary>Negative labels ({negativeLabels.length})</summary>
          <div className="negative-list">
            {negativeLabels.map((label) => (
              <ScoreBar
                key={label}
                label={label}
                displayName={displayNames[label] ?? label}
                score={scores[label]}
                positive={false}
              />
            ))}
          </div>
        </details>
      ) : null}
    </div>
  );
}
