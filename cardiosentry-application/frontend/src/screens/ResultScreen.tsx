import { useEffect, useMemo, useState } from "react";
import { api, friendlyMessage, isApiError } from "../api/client";
import type { CaptureRecord, PredictionResponse } from "../api/types";
import { useApp } from "../AppContext";
import ExplainPanel from "../components/ExplainPanel";
import PredictionLabels from "../components/PredictionLabels";
import StageStrip from "../components/StageStrip";

export interface ResultScreenProps {
  prediction: PredictionResponse;
  onNewCapture: () => void;
  onHistory: () => void;
  onOpenDetail: (id: string) => void;
}

export default function ResultScreen({
  prediction,
  onNewCapture,
  onHistory,
  onOpenDetail,
}: ResultScreenProps) {
  const { config } = useApp();
  const [record, setRecord] = useState<CaptureRecord | null>(null);
  const [recordLoading, setRecordLoading] = useState(true);
  const [recordError, setRecordError] = useState<string | null>(null);
  const captureId = prediction.capture_id;

  useEffect(() => {
    let cancelled = false;
    setRecordLoading(true);
    setRecordError(null);
    api
      .getCapture(captureId)
      .then((r) => {
        if (!cancelled) setRecord(r);
      })
      .catch((e) => {
        if (!cancelled)
          setRecordError(isApiError(e) ? friendlyMessage(e.code, e.message) : String(e));
      })
      .finally(() => {
        if (!cancelled) setRecordLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [captureId]);

  const configLabels = config?.active_model.labels ?? [];
  const displayNames = config?.active_model.display_names ?? {};
  const orderedLabels = useMemo(() => {
    const scoreKeys = Object.keys(prediction.scores);
    const inOrder = configLabels.filter((label) => label in prediction.scores);
    const extra = scoreKeys.filter((label) => !configLabels.includes(label));
    return [...inOrder, ...extra];
  }, [configLabels, prediction.scores]);

  const labelsEntered = !!record?.annotation?.labels_entered_utc;
  const blind = !record || (!!record.session?.blind_mode && !labelsEntered);
  const base = `/api/captures/${captureId}/files`;
  const stages = [
    { key: "original", label: "Original", url: `${base}/original.jpg` },
    { key: "corrected", label: "Corrected", url: `${base}/corrected.png` },
    { key: "model_input", label: "Model input", url: `${base}/model_input.png` },
  ];

  return (
    <div>
      <div className="spread page-heading">
        <h1 className="section-title">Result</h1>
        <span className="muted model-name">{prediction.model.display_name}</span>
      </div>

      {recordError ? <div className="msg warn">{recordError}</div> : null}

      {recordLoading ? (
        <div className="loading-block">Loading result…</div>
      ) : (
        <>
          <PredictionLabels
            labels={orderedLabels}
            displayNames={displayNames}
            scores={prediction.scores}
            positive={prediction.positive}
            blind={blind}
          />

          <ExplainPanel
            captureId={captureId}
            runId={prediction.run_id}
            positive={prediction.positive}
            blind={blind}
          />
        </>
      )}

      <details className="panel compact-details">
        <summary>Images</summary>
        <div className="mt">
          <StageStrip stages={stages} active="model_input" />
        </div>
      </details>

      <div className="footer-nav">
        <button type="button" className="primary" onClick={onNewCapture}>
          New capture
        </button>
        <button type="button" className="ghost" onClick={onHistory}>
          History
        </button>
        <button type="button" className="ghost" onClick={() => onOpenDetail(captureId)}>
          Details
        </button>
      </div>
    </div>
  );
}
