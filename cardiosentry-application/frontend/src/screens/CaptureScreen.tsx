import { useRef, useState } from "react";
import { friendlyMessage, isApiError, uploadCapture } from "../api/client";
import type { CaptureUploadResponse } from "../api/types";
import { useApp } from "../AppContext";

export interface CaptureScreenProps {
  onUploaded: (resp: CaptureUploadResponse) => void;
  onHistory: () => void;
}

export default function CaptureScreen({
  onUploaded,
  onHistory,
}: CaptureScreenProps) {
  const { config, health, loading } = useApp();

  const cameraRef = useRef<HTMLInputElement>(null);
  const galleryRef = useRef<HTMLInputElement>(null);

  const [sheetId, setSheetId] = useState("");
  const [notes, setNotes] = useState("");
  const [retakeOf, setRetakeOf] = useState("");
  const [blindMode, setBlindMode] = useState(false);

  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const blindAvailable = config
    ? config.capture.blind_mode_available !== false
    : true;

  function handleFile(file: File | null, mode: "upload" | "camera") {
    if (!file || uploading) return;
    setUploading(true);
    setError(null);
    setProgress(0);
    uploadCapture({
      file,
      sheetId: sheetId.trim() || undefined,
      notes: notes.trim() || undefined,
      retakeOf: retakeOf.trim() || undefined,
      blindMode: blindMode && blindAvailable,
      mode,
      onProgress: (loaded, total) =>
        setProgress(total > 0 ? loaded / total : 0),
    })
      .then(onUploaded)
      .catch((e) => {
        setError(isApiError(e) ? friendlyMessage(e.code, e.message) : String(e));
      })
      .finally(() => {
        setUploading(false);
        if (cameraRef.current) cameraRef.current.value = "";
        if (galleryRef.current) galleryRef.current.value = "";
      });
  }

  function renderStatus() {
    if (loading && !config && !health) {
      return (
        <span className="status-badge">
          <span className="spinner" /> Connecting…
        </span>
      );
    }
    if (!config || !health) {
      return <span className="status-badge err">backend unreachable</span>;
    }
    const ok = health.model_loaded;
    return (
      <span className="status-badge ok">
        <span className={`dot${ok ? "" : " off"}`} />
        <span>
          {config.active_model.display_name}
          <span className="mono"> · {config.active_model.checkpoint_sha256_short}</span>
          <span className="muted"> · {health.device}</span>
        </span>
      </span>
    );
  }

  return (
    <div>
      <h1 className="section-title">Capture an ECG</h1>
      <p className="section-sub">
        Photograph a printed 12-lead ECG sheet. The app will find the paper
        edges, then you can fine-tune the four corners.
      </p>

      <div className="panel">
        <div className="spread">
          {renderStatus()}
          {health && health.free_space_warn ? (
            <span className="msg warn" style={{ margin: 0 }}>
              Low disk space ({health.free_space_gb} GB free)
            </span>
          ) : null}
        </div>
      </div>

      <div className="big-actions">
        <button
          type="button"
          className="primary"
          disabled={uploading}
          onClick={() => cameraRef.current?.click()}
        >
          Take photo
        </button>
        <button
          type="button"
          disabled={uploading}
          onClick={() => galleryRef.current?.click()}
        >
          Choose photo
        </button>
      </div>

      {/* hidden native inputs (hand off to camera app — works over plain http LAN) */}
      <input
        ref={cameraRef}
        type="file"
        accept="image/*"
        capture="environment"
        hidden
        onChange={(e) => handleFile(e.target.files?.[0] ?? null, "camera")}
      />
      <input
        ref={galleryRef}
        type="file"
        accept="image/*"
        hidden
        onChange={(e) => handleFile(e.target.files?.[0] ?? null, "upload")}
      />

      {uploading ? (
        <div className="panel mt">
          <div className="hstack">
            <span className="spinner" />
            <span>Uploading… {Math.round(progress * 100)}%</span>
          </div>
          <div className="progress">
            <div style={{ width: `${progress * 100}%` }} />
          </div>
        </div>
      ) : null}

      {error ? <div className="msg error mt">{error}</div> : null}

      <div className="panel mt">
        <h3>Capture details</h3>
        <div className="vstack">
          <div className="field">
            <label htmlFor="sheet-id">Sheet ID</label>
            <input
              id="sheet-id"
              type="text"
              value={sheetId}
              onChange={(e) => setSheetId(e.target.value)}
              placeholder="Optional — e.g. patient/study code"
            />
          </div>
          <div className="field">
            <label htmlFor="retake-of">Retake of</label>
            <input
              id="retake-of"
              type="text"
              value={retakeOf}
              onChange={(e) => setRetakeOf(e.target.value)}
              placeholder="Optional — capture ID this replaces"
            />
          </div>
          <div className="field">
            <label htmlFor="notes">Notes</label>
            <textarea
              id="notes"
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              placeholder="Optional — device, paper size, conditions…"
            />
          </div>

          {blindAvailable ? (
            <div className="toggle-row">
              <div className="labels">
                <span className="t">Blind mode</span>
                <span className="d">
                  Hide scores until you enter reference labels first.
                </span>
              </div>
              <span className="switch">
                <input
                  type="checkbox"
                  id="blind-mode"
                  checked={blindMode}
                  onChange={(e) => setBlindMode(e.target.checked)}
                />
                <span className="track" />
              </span>
            </div>
          ) : null}
        </div>
      </div>

      <div className="footer-nav">
        <button type="button" className="ghost" onClick={onHistory}>
          History
        </button>
      </div>
    </div>
  );
}
