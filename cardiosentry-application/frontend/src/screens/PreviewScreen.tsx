import { useState } from "react";
import { api, friendlyMessage, isApiError } from "../api/client";
import type { PredictionResponse, RectifyResponse } from "../api/types";
import { useApp } from "../AppContext";

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

      <div className="preview-grid">
        <div className="preview-cell">
          <img src={rectify.files.corrected} alt="Corrected sheet" />
          <div className="cell-cap">Corrected</div>
        </div>
        <div className="preview-cell">
          <img src={rectify.files.model_input} alt="Model input" />
          <div className="cell-cap exact">Model input</div>
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
      </div>
    </div>
  );
}
