import { useCallback, useEffect, useRef, useState } from "react";
import { api, friendlyMessage, isApiError } from "../api/client";
import type {
  ExplainResponse,
  ExplainSummary,
  LabelAttribution,
} from "../api/types";
import { useApp } from "../AppContext";

export interface ExplainPanelProps {
  captureId: string;
  runId?: string | null;
  positive?: string[];
  blind?: boolean;
}

export default function ExplainPanel({
  captureId,
  runId,
  positive = [],
  blind = false,
}: ExplainPanelProps) {
  const { config } = useApp();
  const labels = config?.active_model.labels ?? [];
  const displayNames = config?.active_model.display_names ?? {};
  const enabled =
    !!config?.explain?.enabled && config.explain.methods.includes("gradcam");

  const [runs, setRuns] = useState<ExplainSummary[]>([]);
  const [current, setCurrent] = useState<ExplainResponse | null>(null);
  const [showOverlay, setShowOverlay] = useState(true);
  const [busy, setBusy] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const refreshRuns = useCallback(async () => {
    try {
      const response = await api.listExplanations(captureId);
      setRuns(response.explanations.filter((run) => run.method === "gradcam"));
    } catch {
      setRuns([]);
    }
  }, [captureId]);

  useEffect(() => {
    void refreshRuns();
  }, [refreshRuns]);

  const timer = useRef<number | null>(null);
  useEffect(() => {
    if (!busy) {
      if (timer.current) window.clearInterval(timer.current);
      timer.current = null;
      return;
    }
    setElapsed(0);
    timer.current = window.setInterval(
      () => setElapsed((value) => value + 1),
      1000
    );
    return () => {
      if (timer.current) window.clearInterval(timer.current);
    };
  }, [busy]);

  async function runGradCam() {
    setBusy(true);
    setError(null);
    try {
      const result = await api.explain(captureId, {
        method: "gradcam",
        labels: positive.length ? positive : labels,
        run_id: runId ?? null,
      });
      setCurrent(result);
      await refreshRuns();
    } catch (e) {
      setError(isApiError(e) ? friendlyMessage(e.code, e.message) : String(e));
    } finally {
      setBusy(false);
    }
  }

  async function openRun(explainId: string) {
    setError(null);
    try {
      const result = await api.getExplanation(captureId, explainId);
      if (result.method === "gradcam") setCurrent(result);
    } catch (e) {
      setError(isApiError(e) ? friendlyMessage(e.code, e.message) : String(e));
    }
  }

  async function removeRun(explainId: string) {
    try {
      await api.deleteExplanation(captureId, explainId);
      if (current?.explain_id === explainId) setCurrent(null);
      await refreshRuns();
    } catch (e) {
      setError(isApiError(e) ? friendlyMessage(e.code, e.message) : String(e));
    }
  }

  if (!enabled) return null;

  return (
    <div className="panel explain-panel">
      <div className="spread explain-title">
        <h2>Grad-CAM</h2>
        {!blind ? (
          <button
            type="button"
            className="small primary"
            onClick={runGradCam}
            disabled={busy}
          >
            {busy ? `Generating… ${elapsed}s` : current ? "Generate again" : "Generate"}
          </button>
        ) : null}
      </div>

      {blind ? <p className="muted compact-copy">Available after labels are entered.</p> : null}
      {error ? <div className="msg error mt">{error}</div> : null}

      {!blind && runs.length ? (
        <div className="run-list" aria-label="Saved Grad-CAM runs">
          {runs.map((run, index) => (
            <div
              key={run.explain_id}
              className={`run-chip${current?.explain_id === run.explain_id ? " active" : ""}${
                run.stale ? " stale" : ""
              }`}
            >
              <button type="button" onClick={() => openRun(run.explain_id)}>
                {index + 1}{run.stale ? " · stale" : ""}
              </button>
              <button
                type="button"
                className="x"
                aria-label={`Delete Grad-CAM run ${index + 1}`}
                onClick={() => removeRun(run.explain_id)}
              >
                ×
              </button>
            </div>
          ))}
        </div>
      ) : null}

      {!blind && current ? (
        <GradCamResult
          result={current}
          displayNames={displayNames}
          showOverlay={showOverlay}
          onToggleOverlay={() => setShowOverlay((value) => !value)}
        />
      ) : null}
    </div>
  );
}

function GradCamResult({
  result,
  displayNames,
  showOverlay,
  onToggleOverlay,
}: {
  result: ExplainResponse;
  displayNames: Record<string, string>;
  showOverlay: boolean;
  onToggleOverlay: () => void;
}) {
  const positiveAttributions = result.attributions.filter(
    (attribution) => attribution.positive !== false
  );
  const negativeAttributions = result.attributions.filter(
    (attribution) => attribution.positive === false
  );

  return (
    <div className="explain-result">
      <div className="explain-result-head">
        <span className="muted">{showOverlay ? "Overlay" : "Heatmap"}</span>
        <button type="button" className="small ghost" onClick={onToggleOverlay}>
          {showOverlay ? "Show heatmap" : "Show overlay"}
        </button>
      </div>
      <AttributionGrid
        attributions={positiveAttributions}
        displayNames={displayNames}
        showOverlay={showOverlay}
      />
      {negativeAttributions.length ? (
        <details className="negative-labels">
          <summary>Negative maps ({negativeAttributions.length})</summary>
          <div className="mt">
            <AttributionGrid
              attributions={negativeAttributions}
              displayNames={displayNames}
              showOverlay={showOverlay}
            />
          </div>
        </details>
      ) : null}
    </div>
  );
}

function AttributionGrid({
  attributions,
  displayNames,
  showOverlay,
}: {
  attributions: LabelAttribution[];
  displayNames: Record<string, string>;
  showOverlay: boolean;
}) {
  return (
    <div className="attr-grid">
      {attributions.map((attribution) => (
        <AttributionCard
          key={attribution.label}
          attribution={attribution}
          displayName={displayNames[attribution.label] ?? attribution.label}
          showOverlay={showOverlay}
        />
      ))}
    </div>
  );
}

function AttributionCard({
  attribution,
  displayName,
  showOverlay,
}: {
  attribution: LabelAttribution;
  displayName: string;
  showOverlay: boolean;
}) {
  const url = showOverlay
    ? attribution.files.overlay
    : attribution.files.heatmap;
  return (
    <figure className="attr-card">
      {url ? (
        <img src={url} alt={`${displayName} Grad-CAM`} loading="lazy" />
      ) : null}
      <figcaption className="attr-head">
        <span className="attr-name">{displayName}</span>
        {typeof attribution.score === "number" ? (
          <span className="score-value small">{attribution.score.toFixed(3)}</span>
        ) : null}
      </figcaption>
    </figure>
  );
}
