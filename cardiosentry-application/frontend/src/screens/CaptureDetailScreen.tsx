import { useEffect, useMemo, useState } from "react";
import { api, friendlyMessage, isApiError } from "../api/client";
import type { CaptureRecord, PredictionRecord } from "../api/types";
import { useApp } from "../AppContext";
import ExplainPanel from "../components/ExplainPanel";
import { formatInputSize } from "../components/ModelBadge";
import PredictionLabels from "../components/PredictionLabels";
import ReferenceLabels from "../components/ReferenceLabels";
import StageStrip from "../components/StageStrip";

export interface CaptureDetailScreenProps {
  captureId: string;
  onBack: () => void;
  onDeleted: () => void;
}

function RunView({
  run,
  fallbackDisplayNames,
  blind,
}: {
  run: PredictionRecord;
  fallbackDisplayNames: Record<string, string>;
  blind: boolean;
}) {
  const scoreLabels = Object.keys(run.scores);
  const labels = (run.descriptor_snapshot?.labels ?? scoreLabels).filter(
    (label) => label in run.scores
  );
  const displayNames = run.descriptor_snapshot?.display_names ?? fallbackDisplayNames;
  return (
    <div className="panel">
      <h2>{run.model.display_name}</h2>
      <PredictionLabels
        labels={labels}
        displayNames={displayNames}
        scores={run.scores}
        positive={run.positive}
        blind={blind}
      />
    </div>
  );
}

export default function CaptureDetailScreen({
  captureId,
  onBack,
  onDeleted,
}: CaptureDetailScreenProps) {
  const { config, models } = useApp();

  const [record, setRecord] = useState<CaptureRecord | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [runModelId, setRunModelId] = useState<string>("");
  const [runningModel, setRunningModel] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  const configDisplayNames = config?.active_model.display_names ?? {};

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    api
      .getCapture(captureId)
      .then((r) => {
        if (!cancelled) setRecord(r);
      })
      .catch((e) => {
        if (!cancelled)
          setError(isApiError(e) ? friendlyMessage(e.code, e.message) : String(e));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [captureId]);

  const runModelIds = useMemo(
    () => new Set((record?.prediction_runs ?? []).map((r) => r.model.id)),
    [record]
  );
  const availableModels = useMemo(
    () => (models?.models ?? []).filter((m) => !runModelIds.has(m.id)),
    [models, runModelIds]
  );

  // The run the attribution maps should explain: the newest one from the model
  // the panel will actually call (the active model), so the scores shown beside
  // a map came from the same checkpoint that produced it.
  const latestRun = useMemo(() => {
    const runs = record?.prediction_runs ?? [];
    const activeId = config?.active_model.id;
    const mine = runs.filter((r) => r.model.id === activeId);
    return (mine.length ? mine : runs)[Math.max(0, (mine.length ? mine : runs).length - 1)];
  }, [record, config]);

  // Labels for the annotation panel: prefer config, fall back to any run.
  const annotationLabels = useMemo(() => {
    if (config?.active_model.labels?.length) return config.active_model.labels;
    const run = record?.prediction_runs?.[0];
    if (run?.descriptor_snapshot?.labels?.length)
      return run.descriptor_snapshot.labels.filter((label) => label in run.scores);
    return run ? Object.keys(run.scores) : [];
  }, [config, record]);

  async function runModel() {
    if (!runModelId) return;
    setRunningModel(true);
    setError(null);
    try {
      await api.predict(captureId, { model_id: runModelId });
      await api.getCapture(captureId).then(setRecord);
    } catch (e) {
      setError(isApiError(e) ? friendlyMessage(e.code, e.message) : String(e));
    } finally {
      setRunningModel(false);
    }
  }

  async function deleteCapture() {
    if (!window.confirm("Delete this capture? It moves to the trash on the server."))
      return;
    setDeleting(true);
    setDeleteError(null);
    try {
      await api.deleteCapture(captureId);
      onDeleted();
    } catch (e) {
      setDeleteError(isApiError(e) ? friendlyMessage(e.code, e.message) : String(e));
      setDeleting(false);
    }
  }

  if (loading) {
    return <div className="loading-block">Loading capture…</div>;
  }
  if (error && !record) {
    return (
      <div>
        <div className="msg error">{error}</div>
        <div className="footer-nav">
          <button type="button" className="ghost" onClick={onBack}>
            Back to history
          </button>
        </div>
      </div>
    );
  }
  if (!record) return null;

  const base = `/api/captures/${captureId}/files`;
  const stages = [
    { key: "original", label: "Original", url: `${base}/original.jpg` },
    { key: "preview", label: "Preview", url: `${base}/preview.jpg` },
    { key: "overlay", label: "Detection overlay", url: `${base}/overlay.jpg` },
    {
      key: "corrected",
      label: "Corrected",
      url: record.correction ? `${base}/corrected.png` : null,
    },
    {
      key: "model_input",
      label: `Model input${
        formatInputSize(record.preprocess?.input_size) ? ` (${formatInputSize(record.preprocess?.input_size)})` : ""
      }`,
      url: record.preprocess ? `${base}/model_input.png` : null,
    },
  ];

  return (
    <div>
      <div className="spread">
        <h1 className="section-title" style={{ marginBottom: 0 }}>
          Capture details
        </h1>
        <button type="button" className="ghost small" onClick={onBack}>
          ← History
        </button>
      </div>
      <p className="capture-id mono">{captureId}</p>

      {error ? <div className="msg error">{error}</div> : null}
      {deleteError ? <div className="msg error">{deleteError}</div> : null}

      <div className="panel">
        <h2>Images</h2>
        <StageStrip stages={stages} />
      </div>

      <details className="panel compact-details">
        <summary>Processing details</summary>
        <div className="kv-grid mt">
          <div className="k">Detection</div>
          <div className="v">{record.detection?.method ?? "—"}</div>
          <div className="k">Confidence</div>
          <div className="v">{record.detection?.confidence ?? "—"}</div>
        </div>
        {record.correction ? (
          <div className="kv-grid">
            <div className="k">Geometry mode</div>
            <div className="v">{record.correction.geometry_mode}</div>
            <div className="k">Mean corner shift</div>
            <div className="v">
              {record.correction.mean_corner_shift_px == null
                ? "—"
                : `${record.correction.mean_corner_shift_px} px`}
            </div>
            <div className="k">Manually adjusted</div>
            <div className="v">
              {record.correction.corners_manually_adjusted ? "yes" : "no"}
            </div>
          </div>
        ) : null}
        {record.preprocess ? (
          <div className="kv-grid">
            <div className="k">Input size</div>
            <div className="v">
              {formatInputSize(record.preprocess.input_size) ?? "—"}
            </div>
            <div className="k">Tensor shape</div>
            <div className="v">
              {JSON.stringify(record.preprocess.tensor_shape ?? "—")}
            </div>
            <div className="k">Tensor sha256</div>
            <div className="v">{String(record.preprocess.tensor_sha256 ?? "—")}</div>
          </div>
        ) : null}
      </details>

      <details className="panel compact-details">
        <summary>Run another model</summary>
        <div className="mt">
        {availableModels.length ? (
          <>
            <div className="field">
              <select
                value={runModelId}
                onChange={(e) => setRunModelId(e.target.value)}
              >
                <option value="">— select a model —</option>
                {availableModels.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.display_name}
                  </option>
                ))}
              </select>
            </div>
            <div className="mt">
              <button
                type="button"
                className="primary"
                onClick={runModel}
                disabled={!runModelId || runningModel}
              >
                {runningModel ? (
                  <>
                    <span className="spinner" /> Running…
                  </>
                ) : (
                  "Run model"
                )}
              </button>
            </div>
          </>
        ) : (
          <p className="muted">All discovered models have been run on this capture.</p>
        )}
        </div>
      </details>

      <h2 style={{ margin: "0 0 10px" }}>
        Prediction runs ({record.prediction_runs?.length ?? 0})
      </h2>
      {(record.prediction_runs ?? []).length === 0 ? (
        <div className="panel muted">No prediction runs yet.</div>
      ) : (
        (record.prediction_runs ?? []).map((run) => (
          <RunView
            key={run.run_id}
            run={run}
            fallbackDisplayNames={configDisplayNames}
            blind={
              !!record.session?.blind_mode && !record.annotation?.labels_entered_utc
            }
          />
        ))
      )}

      {record.preprocess ? (
        <ExplainPanel
          captureId={captureId}
          runId={latestRun?.run_id ?? null}
          positive={latestRun?.positive ?? []}
          blind={
            !!record.session?.blind_mode && !record.annotation?.labels_entered_utc
          }
        />
      ) : null}

      <details className="compact-details mt-lg">
        <summary>Reference labels</summary>
        <div className="mt">
          <ReferenceLabels
            captureId={captureId}
            labels={annotationLabels}
            displayNames={configDisplayNames}
            annotation={record.annotation}
            blindMode={!!record.session?.blind_mode}
            onSaved={(r) => setRecord(r)}
          />
        </div>
      </details>

      <details className="panel">
        <summary>Raw capture.json</summary>
        <pre className="json-view mt">{JSON.stringify(record, null, 2)}</pre>
      </details>

      <div className="footer-nav">
        <button
          type="button"
          className="danger"
          onClick={deleteCapture}
          disabled={deleting}
        >
          {deleting ? "Deleting…" : "Delete capture"}
        </button>
        <button type="button" className="ghost" onClick={onBack}>
          Back to history
        </button>
      </div>
    </div>
  );
}
