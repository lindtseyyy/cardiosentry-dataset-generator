import { useMemo, useState } from "react";
import { api, friendlyMessage, isApiError } from "../api/client";
import type { CaptureUploadResponse, Point, RectifyResponse } from "../api/types";
import { useApp } from "../AppContext";
import QuadEditor from "../components/QuadEditor";
import QualityBanner from "../components/QualityBanner";

const FULL_FRAME: Point[] = [
  [0.05, 0.05],
  [0.95, 0.05],
  [0.95, 0.95],
  [0.05, 0.95],
];

export interface CornerEditScreenProps {
  upload: CaptureUploadResponse;
  onBack: () => void;
  onRectified: (rectify: RectifyResponse) => void;
}

export default function CornerEditScreen({
  upload,
  onBack,
  onRectified,
}: CornerEditScreenProps) {
  const { config } = useApp();

  const initialQuad = useMemo<Point[]>(
    () => upload.detection.quad_norm ?? FULL_FRAME,
    [upload.detection.quad_norm]
  );

  const [quad, setQuad] = useState<Point[]>(initialQuad);
  const [userAdjusted, setUserAdjusted] = useState(false);
  const [geometryMode, setGeometryMode] = useState<string>(
    config?.rectify.default_geometry_mode ?? "training_canvas"
  );
  const [rectifying, setRectifying] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const geometryModes = config?.rectify.geometry_modes ?? ["training_canvas"];
  const confWarn = config?.detection.confidence_warn ?? 0.55;
  const det = upload.detection;

  async function confirm() {
    setRectifying(true);
    setError(null);
    try {
      const rect = await api.rectify(
        upload.capture_id,
        quad,
        userAdjusted ? "manual" : "auto",
        geometryMode
      );
      onRectified(rect);
    } catch (e) {
      setError(isApiError(e) ? friendlyMessage(e.code, e.message) : String(e));
    } finally {
      setRectifying(false);
    }
  }

  return (
    <div>
      <h1 className="section-title">Adjust corners</h1>
      <p className="section-sub">
        Drag the four numbered handles onto the paper corners (TL, TR, BR, BL).
      </p>

      {!det.success ? (
        <div className="msg warn">
          Couldn't find the paper edges — drag the corners yourself.
        </div>
      ) : det.confidence > 0 && det.confidence < confWarn ? (
        <div className="msg warn">
          Automatic detection was unsure — check the corners.
        </div>
      ) : null}

      <QualityBanner quality={upload.quality} />

      <QuadEditor
        imageUrl={upload.files.preview}
        initialQuad={initialQuad}
        onChange={(q, adjusted) => {
          setQuad(q);
          setUserAdjusted(adjusted);
        }}
      />

      <div className="panel mt">
        <div className="field">
          <label htmlFor="geometry-mode">Geometry mode</label>
          <select
            id="geometry-mode"
            value={geometryMode}
            onChange={(e) => setGeometryMode(e.target.value)}
          >
            {geometryModes.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </div>

        {error ? <div className="msg error mt">{error}</div> : null}

        <div className="hstack mt">
          <button
            type="button"
            className="primary"
            onClick={confirm}
            disabled={rectifying}
          >
            {rectifying ? (
              <>
                <span className="spinner" /> Rectifying…
              </>
            ) : (
              "Confirm corners"
            )}
          </button>
          <button type="button" className="ghost" onClick={onBack} disabled={rectifying}>
            Back
          </button>
        </div>
      </div>
    </div>
  );
}
