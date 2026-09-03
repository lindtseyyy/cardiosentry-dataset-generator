import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, friendlyMessage, isApiError } from "../api/client";
import type {
  ExplainConfig,
  ExplainMethod,
  ExplainResponse,
  ExplainSummary,
  IgBaseline,
  LabelAttribution,
} from "../api/types";
import { useApp } from "../AppContext";

/**
 * Attribution maps for one capture: Grad-CAM, Integrated Gradients and the
 * input-output Jacobian, over the same stored model_input.png the scores came
 * from.
 *
 * Three decisions this component exists to enforce:
 *
 *  1. THE COST IS QUOTED BEFORE THE WAIT. Integrated Gradients is `steps` full
 *     forward+backward passes per label on CPU — minutes, not seconds. The
 *     estimate comes from /api/config, and the run button says how many passes
 *     it is about to spend rather than discovering it afterwards.
 *  2. THE HEADER CHECK IS NOT OPTIONAL. Every map is shown with the fraction
 *     of its mass that fell on the printed ID/Age/Sex block, because the record
 *     ID there separates STEMI exactly in the training corpus. A map that looks
 *     convincing and lands on the header is the failure this feature exists to
 *     surface, so it is a badge on the picture, not a number in a details pane.
 *  3. A MAP IS NEVER SHOWN WITHOUT ITS SCORE. An identical heat map means
 *     something different at 0.97 than at 0.11.
 */

const METHOD_LABEL: Record<ExplainMethod, string> = {
  gradcam: "Grad-CAM",
  integrated_gradients: "Integrated Gradients",
  jacobian: "Jacobian",
};

const METHOD_BLURB: Record<ExplainMethod, string> = {
  gradcam:
    "Coarse regions, from the last feature map. Points at a lead, not at a segment within a beat. Seconds.",
  integrated_gradients:
    "Pixel resolution and signed — red raised the score, blue lowered it. Reports whether it converged. Minutes.",
  jacobian:
    "Every label's input gradient at once, plus how much the four labels share evidence. Tens of seconds.",
};

function fmt(n: number | null | undefined, digits = 3): string {
  return typeof n === "number" && Number.isFinite(n) ? n.toFixed(digits) : "—";
}

/** Passes -> a rough wall-clock band. Deliberately coarse: the point is to
 *  distinguish "press it" from "go and do something else". */
function costLabel(passes: number): string {
  if (passes <= 6) return "seconds";
  if (passes <= 20) return "under a minute";
  if (passes <= 60) return "a few minutes";
  return "many minutes";
}

export interface ExplainPanelProps {
  captureId: string;
  /** Prediction run these maps should explain, when the caller has one. */
  runId?: string | null;
  /** Labels the model flagged — the default selection for Integrated Gradients. */
  positive?: string[];
  /** Blind mode with no reference labels yet: scores stay hidden on the maps too. */
  blind?: boolean;
}

export default function ExplainPanel({
  captureId,
  runId,
  positive = [],
  blind = false,
}: ExplainPanelProps) {
  const { config } = useApp();
  const ex: ExplainConfig | undefined = config?.explain;
  const labels = config?.active_model.labels ?? [];
  const displayNames = config?.active_model.display_names ?? {};

  const [method, setMethod] = useState<ExplainMethod>("gradcam");
  const [selected, setSelected] = useState<string[]>([]);
  const [steps, setSteps] = useState<number>(ex?.default_ig_steps ?? 16);
  const [baseline, setBaseline] = useState<IgBaseline>(
    ex?.default_ig_baseline ?? "white"
  );
  const [showOverlay, setShowOverlay] = useState(true);

  const [runs, setRuns] = useState<ExplainSummary[]>([]);
  const [current, setCurrent] = useState<ExplainResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState<string | null>(null);

  // Default selection follows the method's cost: the cheap methods explain the
  // whole head (which is what makes four maps comparable), Integrated
  // Gradients explains only what the model flagged.
  useEffect(() => {
    if (method === "integrated_gradients") {
      setSelected(positive.length ? positive : labels.slice(0, 1));
    } else {
      setSelected(labels);
    }
  }, [method, labels.join(","), positive.join(",")]);

  const refreshRuns = useCallback(async () => {
    try {
      const l = await api.listExplanations(captureId);
      setRuns(l.explanations);
    } catch {
      setRuns([]);
    }
  }, [captureId]);

  useEffect(() => {
    void refreshRuns();
  }, [refreshRuns]);

  // A live timer, not a bare spinner: a five-minute request with no moving
  // number is indistinguishable from a hung one.
  const timer = useRef<number | null>(null);
  useEffect(() => {
    if (!busy) {
      if (timer.current) window.clearInterval(timer.current);
      timer.current = null;
      return;
    }
    setElapsed(0);
    timer.current = window.setInterval(() => setElapsed((e) => e + 1), 1000);
    return () => {
      if (timer.current) window.clearInterval(timer.current);
    };
  }, [busy]);

  const passes = useMemo(() => {
    const c = ex?.cost?.[method];
    if (!c) return 0;
    const n = Math.max(1, selected.length);
    if (method === "integrated_gradients") return steps + 2 + steps * n;
    return c.forward + c.backward_full * n + c.backward_partial * n * 0.15;
  }, [ex, method, selected.length, steps]);

  async function runExplain() {
    setBusy(true);
    setError(null);
    try {
      const body = {
        method,
        labels: selected.length ? selected : null,
        run_id: runId ?? null,
        ...(method === "integrated_gradients" ? { steps, baseline } : {}),
      };
      const r = await api.explain(captureId, body);
      setCurrent(r);
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
      setCurrent(await api.getExplanation(captureId, explainId));
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

  if (!ex) return null;
  if (!ex.enabled) {
    return (
      <div className="panel">
        <h2>Where the model looked</h2>
        <p className="sub">
          Explainability is switched off on this server.
        </p>
      </div>
    );
  }

  return (
    <div className="panel explain-panel">
      <h2>Where the model looked</h2>
      <p className="sub">
        Attribution over the same <span className="mono">model_input.png</span>{" "}
        the scores came from. A map shows what mattered to this model — it is
        not a clinical finding, and a confidently wrong score has one too.
      </p>

      {/* ---------------------------------------------------------- method */}
      <div className="method-tabs" role="tablist" aria-label="Attribution method">
        {ex.methods.map((m) => (
          <button
            key={m}
            type="button"
            role="tab"
            aria-selected={method === m}
            className={`method-tab${method === m ? " active" : ""}`}
            onClick={() => setMethod(m)}
            disabled={busy}
          >
            {METHOD_LABEL[m]}
          </button>
        ))}
      </div>
      <p className="sub method-blurb">{METHOD_BLURB[method]}</p>

      {/* --------------------------------------------------------- labels */}
      <div className="explain-controls">
        <div className="control-row">
          <span className="control-label">Labels</span>
          <div className="chip-row">
            {labels.map((l) => {
              const on = selected.includes(l);
              return (
                <button
                  key={l}
                  type="button"
                  className={`chip toggle${on ? " on" : ""}`}
                  aria-pressed={on}
                  disabled={busy}
                  onClick={() =>
                    setSelected((s) =>
                      s.includes(l) ? s.filter((x) => x !== l) : [...s, l]
                    )
                  }
                >
                  {l}
                </button>
              );
            })}
          </div>
        </div>

        {method === "integrated_gradients" ? (
          <>
            <div className="control-row">
              <label className="control-label" htmlFor="ig-steps">
                Steps
              </label>
              <input
                id="ig-steps"
                type="range"
                min={2}
                max={ex.max_ig_steps}
                step={2}
                value={steps}
                disabled={busy}
                onChange={(e) => setSteps(Number(e.target.value))}
              />
              <span className="val mono">{steps}</span>
            </div>
            <div className="control-row">
              <label className="control-label" htmlFor="ig-baseline">
                Baseline
              </label>
              <select
                id="ig-baseline"
                value={baseline}
                disabled={busy}
                onChange={(e) => setBaseline(e.target.value as IgBaseline)}
              >
                {ex.ig_baselines.map((b) => (
                  <option key={b} value={b}>
                    {b}
                  </option>
                ))}
              </select>
              <span className="sub">
                {baseline === "white"
                  ? "a blank sheet — the reference with no trace on it"
                  : baseline === "blur"
                  ? "the same sheet with the trace destroyed, paper kept"
                  : `a flat ${baseline} image`}
              </span>
            </div>
          </>
        ) : null}
      </div>

      {error ? <div className="msg error">{error}</div> : null}

      <button
        type="button"
        className="primary"
        onClick={runExplain}
        disabled={busy || selected.length === 0}
      >
        {busy ? (
          <>
            <span className="spinner" /> Attributing… {elapsed}s
          </>
        ) : (
          `Run ${METHOD_LABEL[method]}`
        )}
      </button>
      <p className="sub cost-line">
        ≈ {Math.round(passes)} model passes on {config?.active_model.id} —{" "}
        {costLabel(passes)}.{" "}
        {method === "integrated_gradients"
          ? "Each step is a full forward and backward per label."
          : null}
      </p>

      {/* ------------------------------------------------------- past runs */}
      {runs.length ? (
        <div className="run-list">
          {runs.map((r) => (
            <div
              key={r.explain_id}
              className={`run-chip${
                current?.explain_id === r.explain_id ? " active" : ""
              }${r.stale ? " stale" : ""}`}
            >
              <button type="button" onClick={() => openRun(r.explain_id)}>
                {METHOD_LABEL[r.method]}
                <span className="mono"> {r.created_utc.slice(11, 19)}</span>
                {r.stale ? <span className="chip warn">stale</span> : null}
              </button>
              <button
                type="button"
                className="x"
                aria-label={`Delete ${r.explain_id}`}
                onClick={() => removeRun(r.explain_id)}
              >
                ×
              </button>
            </div>
          ))}
        </div>
      ) : null}

      {current ? (
        <ExplainResult
          result={current}
          displayNames={displayNames}
          showOverlay={showOverlay}
          onToggleOverlay={() => setShowOverlay((v) => !v)}
          blind={blind}
        />
      ) : null}
    </div>
  );
}

// ===================================================================== result
function ExplainResult({
  result,
  displayNames,
  showOverlay,
  onToggleOverlay,
  blind,
}: {
  result: ExplainResponse;
  displayNames: Record<string, string>;
  showOverlay: boolean;
  onToggleOverlay: () => void;
  blind: boolean;
}) {
  const coupling = result.extras?.coupling;
  const signed = result.attributions.some((a) => a.signed);

  return (
    <div className="explain-result">
      <div className="explain-result-head">
        <div>
          <strong>{METHOD_LABEL[result.method]}</strong>{" "}
          <span className="sub">
            {result.timing_ms.attribute
              ? `${(result.timing_ms.attribute / 1000).toFixed(1)} s`
              : null}
            {typeof result.meta.target_layer === "string"
              ? ` · layer ${result.meta.target_layer}`
              : null}
            {typeof result.meta.steps === "number"
              ? ` · ${result.meta.steps} steps, ${result.meta.baseline} baseline`
              : null}
            {/* Why this run took ~2x as long as the estimate said. */}
            {(result.meta.memory as { checkpointed?: boolean } | undefined)
              ?.checkpointed
              ? " · low-memory mode (recomputed activations)"
              : null}
          </span>
        </div>
        <button type="button" className="small ghost" onClick={onToggleOverlay}>
          {showOverlay ? "Heat map only" : "Overlay on sheet"}
        </button>
      </div>

      {signed ? (
        <div className="legend diverging">
          <span>lowered the score</span>
          <span className="bar" aria-hidden="true" />
          <span>raised it</span>
        </div>
      ) : (
        <div className="legend sequential">
          <span>less</span>
          <span className="bar" aria-hidden="true" />
          <span>more</span>
        </div>
      )}

      <div className="attr-grid">
        {result.attributions.map((a) => (
          <AttributionCard
            key={a.label}
            a={a}
            displayName={displayNames[a.label] ?? a.label}
            showOverlay={showOverlay}
            blind={blind}
          />
        ))}
      </div>

      {coupling ? <CouplingMatrix coupling={coupling} /> : null}

      <ul className="caveats">
        {result.caveats.map((c, i) => (
          <li key={i}>{c}</li>
        ))}
      </ul>
    </div>
  );
}

// ------------------------------------------------------------------ one map
function AttributionCard({
  a,
  displayName,
  showOverlay,
  blind,
}: {
  a: LabelAttribution;
  displayName: string;
  showOverlay: boolean;
  blind: boolean;
}) {
  const url = showOverlay ? a.files.overlay : a.files.heatmap;
  const lift = a.header?.lift ?? null;
  // 2.0 = twice the attention the header's AREA would predict. Below that the
  // band is getting roughly its fair share and the badge stays quiet.
  const headerHot = lift !== null && lift >= 2.0;
  const completeness = a.stats.completeness_delta as number | null | undefined;
  const poorlyConverged =
    typeof completeness === "number" && completeness > 0.2;
  const saturated = a.stats.saturated === true;

  return (
    <figure className="attr-card">
      {url ? <img src={url} alt={`${displayName} attribution map`} loading="lazy" /> : null}
      <figcaption>
        <div className="attr-head">
          <span className="attr-name">{displayName}</span>
          {blind ? (
            <span className="chip muted">hidden</span>
          ) : typeof a.score === "number" ? (
            <span className={`chip ${a.positive ? "pos" : "neg"}`}>
              {a.score.toFixed(3)}
            </span>
          ) : null}
        </div>

        <div className="attr-badges">
          <span
            className={`badge${headerHot ? " danger" : " ok"}`}
            title={
              "Share of attribution mass on the printed ID/Age/Sex block, " +
              "divided by that block's share of the sheet area. 1.0 is its " +
              "fair share; the record ID printed there separates STEMI " +
              "exactly in the training corpus."
            }
          >
            header ×{fmt(lift, 2)}
          </span>
          {poorlyConverged ? (
            <span className="badge warn" title="Attributions do not yet sum to the logit difference — raise the step count.">
              not converged ({(completeness! * 100).toFixed(0)}%)
            </span>
          ) : null}
          {saturated ? (
            <span className="badge warn" title="Near-zero gradient: this logit is saturated and its map is numerical noise.">
              saturated
            </span>
          ) : null}
          {a.summary.mass_half_area_frac !== null ? (
            <span
              className="badge"
              title="Smallest fraction of the sheet holding half the attribution mass. Small = focused, large = diffuse."
            >
              half in {(a.summary.mass_half_area_frac * 100).toFixed(1)}%
            </span>
          ) : null}
        </div>
      </figcaption>
    </figure>
  );
}

// ------------------------------------------------------- coupling (jacobian)
function CouplingMatrix({
  coupling,
}: {
  coupling: Record<string, Record<string, number>>;
}) {
  const keys = Object.keys(coupling);
  return (
    <div className="coupling">
      <h3>Label coupling</h3>
      <p className="sub">
        Cosine between each pair of input-gradient rows. <strong>+1</strong> —
        the two labels move on identical evidence. <strong>−1</strong> — the
        same evidence, read in opposite directions. <strong>0</strong> —
        unrelated pixels.
      </p>
      <div className="table-scroll">
        <table className="coupling-table">
          <thead>
            <tr>
              <th />
              {keys.map((k) => (
                <th key={k}>{k}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {keys.map((r) => (
              <tr key={r}>
                <th>{r}</th>
                {keys.map((c) => {
                  const v = coupling[r][c];
                  // Blue for negative, red for positive, transparent at zero —
                  // the same encoding the signed maps use, so the table and the
                  // pictures read as one thing.
                  const bg =
                    v >= 0
                      ? `rgba(214, 69, 65, ${Math.min(Math.abs(v), 1) * 0.55})`
                      : `rgba(48, 110, 214, ${Math.min(Math.abs(v), 1) * 0.55})`;
                  return (
                    <td key={c} style={{ background: bg }} className="mono">
                      {v.toFixed(2)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
