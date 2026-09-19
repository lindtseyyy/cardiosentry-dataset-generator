import { useEffect, useState } from "react";
import { api, friendlyMessage, isApiError } from "../api/client";
import type { CaptureSummary } from "../api/types";
import { useApp } from "../AppContext";

export interface HistoryScreenProps {
  onOpenCapture: (id: string) => void;
  onNewCapture: () => void;
}

function fmtTime(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString();
}

export default function HistoryScreen({
  onOpenCapture,
  onNewCapture,
}: HistoryScreenProps) {
  const { config } = useApp();
  const [captures, setCaptures] = useState<CaptureSummary[]>([]);
  const [page, setPage] = useState(1);
  const [hasMore, setHasMore] = useState(false);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const displayNames = config?.active_model.display_names ?? {};

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    api
      .listCaptures(1)
      .then((r) => {
        if (cancelled) return;
        setCaptures(r.captures);
        setPage(r.page);
        setHasMore(r.has_more);
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
  }, []);

  async function loadMore() {
    setLoadingMore(true);
    setError(null);
    try {
      const r = await api.listCaptures(page + 1);
      setCaptures((prev) => [...prev, ...r.captures]);
      setPage(r.page);
      setHasMore(r.has_more);
    } catch (e) {
      setError(isApiError(e) ? friendlyMessage(e.code, e.message) : String(e));
    } finally {
      setLoadingMore(false);
    }
  }

  return (
    <div>
      <h1 className="section-title">History</h1>
      <p className="section-sub">
        {captures.length} capture{captures.length === 1 ? "" : "s"}
      </p>

      {error ? <div className="msg error">{error}</div> : null}

      {loading ? (
        <div className="loading-block">Loading history…</div>
      ) : captures.length === 0 ? (
        <div className="panel center muted">No captures yet.</div>
      ) : (
        <div className="history-list">
          {captures.map((c) => (
            <button
              key={c.capture_id}
              type="button"
              className="history-card"
              onClick={() => onOpenCapture(c.capture_id)}
            >
              <div className="history-thumb">
                {c.thumbnail_url ? (
                  <img src={c.thumbnail_url} alt="" loading="lazy" />
                ) : (
                  <span className="placeholder">ECG</span>
                )}
              </div>
              <div className="history-body">
                <div className="history-title">
                  <span className="mono">{c.capture_id}</span>
                </div>
                <div className="history-meta">{fmtTime(c.created_utc)}</div>
                {c.sheet_id ? (
                  <div className="history-meta">sheet {c.sheet_id}</div>
                ) : null}
                {c.latest_positive.length ? (
                  <div className="hstack" style={{ marginTop: 6 }}>
                    {c.latest_positive.map((l) => (
                      <span key={l} className="chip pos">
                        {displayNames[l] ?? l}
                      </span>
                    ))}
                  </div>
                ) : null}
              </div>
            </button>
          ))}
        </div>
      )}

      {hasMore ? (
        <div className="mt">
          <button
            type="button"
            className="ghost"
            onClick={loadMore}
            disabled={loadingMore}
          >
            {loadingMore ? "Loading…" : "Load more"}
          </button>
        </div>
      ) : null}

      <div className="footer-nav">
        <button type="button" className="primary" onClick={onNewCapture}>
          New capture
        </button>
      </div>
    </div>
  );
}
