import { useEffect, useMemo, useState } from "react";
import { api, friendlyMessage, isApiError } from "../api/client";
import type { CaptureRecord, PredictionResponse } from "../api/types";
import { useApp } from "../AppContext";
import ExplainPanel from "../components/ExplainPanel";
import ModelBadge from "../components/ModelBadge";
import QualityBanner from "../components/QualityBanner";
import ReferenceLabels from "../components/ReferenceLabels";
import ScoreBar from "../components/ScoreBar";
import StageStrip from "../components/StageStrip";

export interface ResultScreenProps {
  prediction: PredictionResponse;
  onPrediction: (p: PredictionResponse) => void;
  onNewCapture: () => void;
  onHistory: () => void;
  onOpenDetail: (id: string) => void;
}

export default function ResultScreen({
  prediction,
  onPrediction,
  onNewCapture,
  onHistory,
  onOpenDetail,
}: ResultScreenProps) {
  const { config } = useApp();

  const [record, setRecord] = useState<CaptureRecord | null>(null);
  const [recordLoading, setRecordLoading] = useState(true);
  const [recordError, setRecordError] = useState<string | null>(null);

  const [sliders, setSliders] = useState<Record<string, number>>(() => ({
    ...prediction.thresholds,
  }));
  const [touched, setTouched] = useState(false);
  const [savingOverride, setSavingOverride] = useState(false);
  const [error, setError] = useState<string | null>(null);

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

  // Reset the threshold sliders whenever a new run lands (e.g. override save).
  useEffect(() => {
    setSliders({ ...prediction.thresholds });
    setTouched(false);
    setError(null);
  }, [prediction.run_id, prediction.thresholds]);

  const configLabels = config?.active_model.labels ?? [];
  const displayNames = config?.active_model.display_names ?? {};
  const scoreKeys = Object.keys(prediction.scores);
  const orderedLabels = useMemo(() => {
    const inOrder = configLabels.filter((l) => l in prediction.scores);
    const extra = scoreKeys.filter((l) => !configLabels.includes(l));
    return [...inOrder, ...extra];
  }, [configLabels, scoreKeys, prediction.scores]);

  const labelsEntered = !!record?.annotation?.labels_entered_utc;
  const blind = !!record?.session?.blind_mode && !labelsEntered;

  const normalKey = orderedLabels.find(
    (l) =>
      l.toUpperCase() === "NORMAL" ||
      (displayNames[l] ?? "").toLowerCase() === "normal"
  );
  const bothPositive =
    !!normalKey &&
    prediction.positive.includes(normalKey) &&
    prediction.positive.some((l) => l !== normalKey);

  const base = `/api/captures/${captureId}/files`;
  const stages = [
    { key: "original", label: "Original", url: `${base}/original.jpg` },
    { key: "corrected", label: "Corrected", url: `${base}/corrected.png` },
    {
      key: "model_input",
      label: "Model input (this result)",
      url: `${base}/model_input.png`,
    },
  ];

  async function saveOverride() {
    setSavingOverride(true);
    setError(null);
    try {
      const p = await api.predict(captureId, {
        model_id: prediction.model.id,
        thresholds: sliders,
      });
      onPrediction(p);
    } catch (e) {
      setError(isApiError(e) ? friendlyMessage(e.code, e.message) : String(e));
    } finally {
      setSavingOverride(false);
    }
  }

  return (
    <div>
      <h1 className="section-title">Result</h1>
      <p className="section-sub">
        Four independent model scores — not a single overall assessment.
      </p>

      <ModelBadge
        id={prediction.model.id}
        displayName={prediction.model.display_name}
        version={prediction.model.version}
        checkpointShort={prediction.model.checkpoint_sha256_short}
        inputSize={prediction.model.input_size}
        device={prediction.model.device}
      />

      <div className="panel mt">
        <div className="kv-grid">
          <div className="k">Preprocess</div>
          <div className="v">{prediction.timing_ms.preprocess} ms</div>
          <div className="k">Forward</div>
          <div className="v">{prediction.timing_ms.forward} ms</div>
        </div>
      </div>

      {bothPositive ? (
        <div className="msg neutral">
          The model flagged both NORMAL and a pathology — both are shown below.
        </div>
      ) : null}

      {recordError ? <div className="msg warn">{recordError}</div> : null}

      {recordLoading ? (
        <div className="loading-block">Loading capture…</div>
      ) : (
        <>
          <QualityBanner quality={record?.quality_model_input} />

          {orderedLabels.map((label) => {
            const slider = sliders[label];
            const effThr =
              typeof slider === "number" ? slider : prediction.thresholds[label];
            // Re-derive Positive/Negative CLIENT-SIDE from the stored score
            // against the (possibly overridden) threshold — never re-runs inference.
            const isPositive = touched
              ? prediction.scores[label] >= effThr
              : prediction.positive.includes(label);
            return (
              <div key={label}>
                <ScoreBar
                  label={label}
                  displayName={displayNames[label] ?? label}
                  score={prediction.scores[label]}
                  threshold={prediction.thresholds[label]}
                  thresholdSource={prediction.threshold_source}
                  positive={isPositive}
                  blind={blind}
                  effectiveThreshold={touched ? effThr : undefined}
                />
                {!blind ? (
                  <div className="threshold-slider">
                    <input
                      type="range"
                      min={0}
                      max={1}
                      step={0.01}
                      aria-label={`Threshold override for ${label}`}
                      value={sliders[label] ?? prediction.thresholds[label]}
                      onChange={(e) => {
                        const v = Number(e.target.value);
                        setSliders((s) => ({ ...s, [label]: v }));
                        setTouched(true);
                      }}
                    />
                    <span className="val">
                      {(sliders[label] ?? prediction.thresholds[label]).toFixed(2)}
                    </span>
                  </div>
                ) : null}
              </div>
            );
          })}

          {touched && !blind ? (
            <div className="panel">
              <p className="sub">
                A manual threshold override. Saving creates a new run marked{" "}
                <span className="mono">user_override</span> — it never re-runs
                inference on the photo.
              </p>
              {error ? <div className="msg error mt">{error}</div> : null}
              <button
                type="button"
                className="primary"
                onClick={saveOverride}
                disabled={savingOverride}
              >
                {savingOverride ? (
                  <>
                    <span className="spinner" /> Saving…
                  </>
                ) : (
                  "Save override run"
                )}
              </button>
            </div>
          ) : null}
        </>
      )}

      <ExplainPanel
        captureId={captureId}
        runId={prediction.run_id}
        positive={prediction.positive}
        blind={blind}
      />

      <div className="panel">
        <h2>Pipeline stages</h2>
        <StageStrip stages={stages} active="model_input" />
      </div>

      {config ? <div className="notice mt">{config.notice}</div> : null}

      {record && !recordLoading ? (
        <div className="mt-lg">
          <ReferenceLabels
            captureId={captureId}
            labels={configLabels.length ? configLabels : orderedLabels}
            displayNames={displayNames}
            annotation={record.annotation}
            blindMode={!!record.session?.blind_mode}
            onSaved={(r) => setRecord(r)}
          />
        </div>
      ) : null}

      <div className="footer-nav">
        <button type="button" className="primary" onClick={onNewCapture}>
          New capture
        </button>
        <button type="button" className="ghost" onClick={onHistory}>
          History
        </button>
        <button
          type="button"
          className="ghost"
          onClick={() => onOpenDetail(captureId)}
        >
          Full details
        </button>
      </div>
    </div>
  );
}
