import { useEffect, useState } from "react";
import { api, friendlyMessage, isApiError } from "../api/client";
import type { AnnotationRecord, CaptureRecord } from "../api/types";

const TRUTH_SOURCES = [
  "cardiologist report",
  "PTB-XL metadata",
  "visual inspection",
  "unknown",
];

export interface ReferenceLabelsProps {
  captureId: string;
  labels: string[];
  displayNames: Record<string, string>;
  annotation: AnnotationRecord;
  blindMode?: boolean;
  onSaved: (record: CaptureRecord) => void;
}

/**
 * Reference-label panel (§17.6): four checkboxes matching the model heads,
 * truth source, notes, and the labels_entered_before_prediction safeguard.
 */
export default function ReferenceLabels({
  captureId,
  labels,
  displayNames,
  annotation,
  blindMode = false,
  onSaved,
}: ReferenceLabelsProps) {
  const [refLabels, setRefLabels] = useState<Record<string, boolean>>(
    annotation.reference_labels ?? {}
  );
  const [truthSource, setTruthSource] = useState<string>(
    annotation.truth_source ?? ""
  );
  const [notes, setNotes] = useState<string>(annotation.notes ?? "");
  const [beforePrediction, setBeforePrediction] = useState<boolean>(
    annotation.labels_entered_before_prediction ?? blindMode
  );
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);

  // Re-sync if the record changes underneath us (e.g. after a save elsewhere).
  useEffect(() => {
    setRefLabels(annotation.reference_labels ?? {});
    setTruthSource(annotation.truth_source ?? "");
    setNotes(annotation.notes ?? "");
    setBeforePrediction(annotation.labels_entered_before_prediction ?? blindMode);
  }, [annotation, blindMode]);

  function toggleLabel(label: string, value: boolean) {
    setRefLabels((prev) => {
      const next = { ...prev };
      if (value) next[label] = true;
      else delete next[label];
      return next;
    });
  }

  async function save() {
    setSaving(true);
    setError(null);
    setSaved(false);
    try {
      const record = await api.patchAnnotation(captureId, {
        reference_labels: refLabels,
        labels_entered_before_prediction: beforePrediction,
        truth_source: truthSource || null,
        notes: notes || null,
      });
      setSaved(true);
      onSaved(record);
    } catch (e) {
      setError(isApiError(e) ? friendlyMessage(e.code, e.message) : String(e));
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="panel">
      <h2>Reference labels</h2>

      <div className="checks-grid">
        {labels.map((label) => (
          <label key={label} className="check">
            <input
              type="checkbox"
              checked={!!refLabels[label]}
              onChange={(e) => toggleLabel(label, e.target.checked)}
            />
            <span>{displayNames[label] ?? label}</span>
          </label>
        ))}
      </div>

      <div className="vstack mt">
        <div className="field">
          <label htmlFor="truth-source">Truth source</label>
          <select
            id="truth-source"
            value={truthSource}
            onChange={(e) => setTruthSource(e.target.value)}
          >
            <option value="">— select —</option>
            {TRUTH_SOURCES.map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
        </div>

        <div className="field">
          <label htmlFor="ref-notes">Notes</label>
          <textarea
            id="ref-notes"
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
            placeholder="Optional"
          />
        </div>

        <div className="toggle-row">
          <div className="labels">
            <span className="t">Labels entered before prediction</span>
          </div>
          <span className="switch">
            <input
              type="checkbox"
              id="before-prediction"
              checked={beforePrediction}
              onChange={(e) => setBeforePrediction(e.target.checked)}
            />
            <span className="track" />
          </span>
        </div>
      </div>

      {error ? <div className="msg error mt">{error}</div> : null}
      {saved ? (
        <div className="msg info mt">Reference labels saved.</div>
      ) : null}

      <div className="mt">
        <button type="button" className="primary" onClick={save} disabled={saving}>
          {saving ? "Saving…" : "Save labels"}
        </button>
      </div>
    </div>
  );
}
