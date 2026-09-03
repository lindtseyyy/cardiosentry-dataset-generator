import { useState } from "react";
import { api, friendlyMessage, isApiError } from "../api/client";
import type { PredictionResponse, RectifyResponse } from "../api/types";
import { useApp } from "../AppContext";
import { formatInputSize } from "../components/ModelBadge";
import QualityBanner from "../components/QualityBanner";

export interface PreviewScreenProps {
  rectify: RectifyResponse;
  onBack: () => void;
  onPredicted: (prediction: PredictionResponse) => void;
}

export default function PreviewScreen({
  rectify,
  onBack,
  onPredicted,
}: PreviewScreenProps) {
  const { config, models } = useApp();
  const [modelId, setModelId] = useState<string>(
    models?.active_id ?? config?.active_model.id ?? ""
  );
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const inputSize = formatInputSize(
    rectify.preprocess?.input_size ?? config?.active_model.input_size
  );

  async function run() {
    setRunning(true);
    setError(null);
    try {
      const prediction = await api.predict(rectify.capture_id, {
        model_id: modelId || null,
      });
      onPredicted(prediction);
    } catch (e) {
      setError(isApiError(e) ? friendlyMessage(e.code, e.message) : String(e));
    } finally {
      setRunning(false);
    }
  }

  return (
    <div>
      <h1 className="section-title">Preview</h1>
      <p className="section-sub">
        The corrected sheet and the exact model input. Re-adjust the corners if
        anything looks wrong.
      </p>

      <QualityBanner quality={rectify.quality} />

      <div className="kv-grid panel" style={{ marginBottom: 16 }}>
        <div className="k">Geometry mode</div>
        <div className="v">{rectify.geometry_mode}</div>
        <div className="k">Corrected size</div>
        <div className="v">
          {rectify.corrected_px[0]} × {rectify.corrected_px[1]} px
        </div>
        <div className="k">Corner shift</div>
        <div className="v">
          {rectify.mean_corner_shift_px == null
            ? "—"
            : `${rectify.mean_corner_shift_px} px`}
        </div>
      </div>

      <div className="preview-grid">
        <div className="preview-cell">
          <img src={rectify.files.corrected} alt="Corrected sheet" />
          <div className="cell-cap">corrected.png</div>
        </div>
        <div className="preview-cell">
          <img src={rectify.files.model_input} alt="Model input" />
          <div className="cell-cap exact">
            model_input.png ({inputSize}) — this is exactly what the model will
            see
          </div>
        </div>
      </div>

      <div className="panel mt">
        <div className="field">
          <label htmlFor="model-select">Model</label>
          <select
            id="model-select"
            value={modelId}
            onChange={(e) => setModelId(e.target.value)}
          >
            {(models?.models ?? []).map((m) => (
              <option key={m.id} value={m.id}>
                {m.display_name}
                {m.active ? " (active)" : ""}
              </option>
            ))}
            {!(models?.models ?? []).length && config?.active_model.id ? (
              <option value={config.active_model.id}>
                {config.active_model.display_name}
              </option>
            ) : null}
          </select>
        </div>

        {error ? <div className="msg error mt">{error}</div> : null}

        <div className="hstack mt">
          <button
            type="button"
            className="primary"
            onClick={run}
            disabled={running}
          >
            {running ? (
              <>
                <span className="spinner" /> Running model…
              </>
            ) : (
              "Run model"
            )}
          </button>
          <button type="button" className="ghost" onClick={onBack} disabled={running}>
            Re-adjust corners
          </button>
        </div>
        {running ? (
          <p className="muted" style={{ fontSize: "0.8rem" }}>
            Inference typically takes 1–2 s (waiting if another request queues).
          </p>
        ) : null}
      </div>
    </div>
  );
}
