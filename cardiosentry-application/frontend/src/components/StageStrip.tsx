import { useEffect, useState } from "react";

export interface Stage {
  key: string;
  label: string;
  url: string | null;
}

export interface StageStripProps {
  stages: Stage[];
  active?: string;
}

export default function StageStrip({ stages, active }: StageStripProps) {
  const [open, setOpen] = useState<Stage | null>(null);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);

  const present = stages.filter((s) => s.url);

  return (
    <>
      <div className="stage-strip">
        {present.map((s) => (
          <div
            key={s.key}
            className={`stage-item${active === s.key ? " active" : ""}`}
            role="button"
            tabIndex={0}
            onClick={() => setOpen(s)}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                setOpen(s);
              }
            }}
          >
            <img src={s.url!} alt={s.label} loading="lazy" />
            <span className="stage-label">{s.label}</span>
          </div>
        ))}
      </div>

      {open && (
        <div className="lightbox" onClick={() => setOpen(null)}>
          <button
            className="close"
            aria-label="Close"
            onClick={(e) => {
              e.stopPropagation();
              setOpen(null);
            }}
          >
            ×
          </button>
          <img
            src={open.url!}
            alt={open.label}
            onClick={(e) => e.stopPropagation()}
          />
          <span className="caption">{open.label}</span>
        </div>
      )}
    </>
  );
}
