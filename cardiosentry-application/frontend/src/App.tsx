import { useCallback, useEffect, useMemo, useState } from "react";
import { api, friendlyMessage, isApiError } from "./api/client";
import { AppContext, type AppData } from "./AppContext";
import type {
  AppConfig,
  CaptureUploadResponse,
  HealthResponse,
  ModelsResponse,
  PredictionResponse,
  RectifyResponse,
} from "./api/types";
import CaptureScreen from "./screens/CaptureScreen";
import CaptureDetailScreen from "./screens/CaptureDetailScreen";
import CornerEditScreen from "./screens/CornerEditScreen";
import HistoryScreen from "./screens/HistoryScreen";
import PreviewScreen from "./screens/PreviewScreen";
import ResultScreen from "./screens/ResultScreen";

type Screen =
  | { name: "capture" }
  | { name: "history" }
  | { name: "detail"; captureId: string }
  | { name: "corner" }
  | { name: "preview"; rectify: RectifyResponse }
  | { name: "result"; prediction: PredictionResponse };

function screenFromHash(): Screen {
  const h = window.location.hash;
  if (/^#\/history$/.test(h)) return { name: "history" };
  const detail = h.match(/^#\/capture\/([A-Za-z0-9_.-]+)$/);
  if (detail) return { name: "detail", captureId: detail[1] };
  return { name: "capture" };
}

export default function App() {
  const [screen, setScreen] = useState<Screen>(() => screenFromHash());
  const [upload, setUpload] = useState<CaptureUploadResponse | null>(null);

  const [config, setConfig] = useState<AppConfig | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [models, setModels] = useState<ModelsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    // Config + health are critical (labels, thresholds, notice, status).
    try {
      const [c, h] = await Promise.all([api.getConfig(), api.getHealth()]);
      setConfig(c);
      setHealth(h);
    } catch (e) {
      setError(isApiError(e) ? friendlyMessage(e.code, e.message) : String(e));
    }
    // Models are best-effort: screens fall back to config.active_model.
    try {
      setModels(await api.getModels());
    } catch {
      setModels(null);
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  useEffect(() => {
    const onHash = () => setScreen(screenFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const ctx = useMemo<AppData>(
    () => ({ config, health, models, loading, error, reload: load }),
    [config, health, models, loading, error, load]
  );

  const goCapture = useCallback(() => {
    setScreen({ name: "capture" });
    window.location.hash = "";
  }, []);

  const goHistory = useCallback(() => {
    setScreen({ name: "history" });
    window.location.hash = "#/history";
  }, []);

  const goDetail = useCallback((id: string) => {
    setScreen({ name: "detail", captureId: id });
    window.location.hash = `#/capture/${id}`;
  }, []);

  function renderScreen() {
    switch (screen.name) {
      case "capture":
        return (
          <CaptureScreen
            onUploaded={(resp) => {
              setUpload(resp);
              setScreen({ name: "corner" });
            }}
            onHistory={goHistory}
          />
        );
      case "history":
        return (
          <HistoryScreen onOpenCapture={goDetail} onNewCapture={goCapture} />
        );
      case "detail":
        return (
          <CaptureDetailScreen
            captureId={screen.captureId}
            onBack={goHistory}
            onDeleted={goHistory}
          />
        );
      case "corner":
        if (!upload) {
          return <div className="loading-block">No capture in progress.</div>;
        }
        return (
          <CornerEditScreen
            upload={upload}
            onBack={goCapture}
            onRectified={(rect) => setScreen({ name: "preview", rectify: rect })}
          />
        );
      case "preview":
        return (
          <PreviewScreen
            rectify={screen.rectify}
            onBack={() => setScreen({ name: "corner" })}
            onPredicted={(p) => setScreen({ name: "result", prediction: p })}
          />
        );
      case "result":
        return (
          <ResultScreen
            prediction={screen.prediction}
            onPrediction={(p) => setScreen({ name: "result", prediction: p })}
            onNewCapture={goCapture}
            onHistory={goHistory}
            onOpenDetail={goDetail}
          />
        );
    }
  }

  const dotClass = !health ? "off" : health.model_loaded ? "" : "warn";

  return (
    <AppContext.Provider value={ctx}>
      <div className="app">
        <div className="topbar">
          <span className="brand">
            <span className={`dot${dotClass}`} />
            CardioSentry
          </span>
          <nav className="hstack" style={{ gap: 8 }}>
            <button type="button" className="small ghost" onClick={goCapture}>
              New
            </button>
            <button type="button" className="small ghost" onClick={goHistory}>
              History
            </button>
          </nav>
        </div>

        {error ? (
          <div className="backend-banner">
            <span>{error}</span>
            <button type="button" onClick={load}>
              Retry
            </button>
          </div>
        ) : null}

        {renderScreen()}
      </div>
    </AppContext.Provider>
  );
}
