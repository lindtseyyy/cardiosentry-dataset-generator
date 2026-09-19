import { useRef, useState } from "react";
import { friendlyMessage, isApiError, uploadCapture } from "../api/client";
import type { CaptureUploadResponse } from "../api/types";
import { useApp } from "../AppContext";

export interface CaptureScreenProps {
  onUploaded: (resp: CaptureUploadResponse) => void;
}

export default function CaptureScreen({ onUploaded }: CaptureScreenProps) {
  const { config, health, loading } = useApp();

  const cameraRef = useRef<HTMLInputElement>(null);
  const galleryRef = useRef<HTMLInputElement>(null);

  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);

  function handleFile(file: File | null, mode: "upload" | "camera") {
    if (!file || uploading) return;
    setUploading(true);
    setError(null);
    setProgress(0);
    uploadCapture({
      file,
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
        <span>{config.active_model.display_name}</span>
      </span>
    );
  }

  return (
    <div>
      <h1 className="section-title page-heading">New capture</h1>

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

    </div>
  );
}
