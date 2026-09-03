import { useEffect, useRef, useState } from "react";
import type { Point } from "../api/types";

/**
 * QuadEditor — canvas editor for the four paper corners.
 *
 * Handles work in NORMALIZED [0,1] coordinates (conversion to pixels happens
 * server-side only). Drag with pointer events (touch-friendly), a loupe
 * magnifier while dragging, clamping to [0,1], and convexity validation that
 * reverts a crossing drag. Includes reset-to-auto / reset-to-full-frame / undo.
 */

interface XY {
  x: number;
  y: number;
}

export interface QuadEditorProps {
  imageUrl: string;
  initialQuad: Point[];
  onChange?: (quad: Point[], userAdjusted: boolean) => void;
}

const HIT_RADIUS = 24; // px — large touch target
const HANDLE_ORDER = ["TL", "TR", "BR", "BL"];

function toXY(points: Point[]): XY[] {
  return points.map(([x, y]) => ({ x, y }));
}
function toPoints(quad: XY[]): Point[] {
  return quad.map((p) => [p.x, p.y]);
}
function cloneQuad(quad: XY[]): XY[] {
  return quad.map((p) => ({ ...p }));
}

/** Convexity via cross-product sign consistency over the 4 corners. */
function isConvex(quad: XY[]): boolean {
  if (quad.length !== 4) return false;
  const signs: number[] = [];
  for (let i = 0; i < 4; i++) {
    const a = quad[i];
    const b = quad[(i + 1) % 4];
    const c = quad[(i + 2) % 4];
    const cross = (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x);
    if (Math.abs(cross) > 1e-9) signs.push(Math.sign(cross));
  }
  if (signs.length === 0) return false;
  return signs.every((s) => s === signs[0]);
}

const FULL_FRAME_INSET: XY[] = [
  { x: 0.05, y: 0.05 },
  { x: 0.95, y: 0.05 },
  { x: 0.95, y: 0.95 },
  { x: 0.05, y: 0.95 },
];

function drawLoupe(
  ctx: CanvasRenderingContext2D,
  img: HTMLImageElement,
  w: number,
  h: number,
  center: XY,
  index: number
) {
  const isTop = index === 0 || index === 1; // TL, TR
  const R = 56;
  const offsetY = isTop ? -R - 16 : R + 16;
  let lx = center.x;
  let ly = center.y + offsetY;
  lx = Math.max(R, Math.min(w - R, lx));
  ly = Math.max(R, Math.min(h - R, ly));

  const zoom = 3;
  const iw = img.naturalWidth;
  const ih = img.naturalHeight;
  const sx0 = (center.x - R / zoom) * (iw / w);
  const sy0 = (center.y - R / zoom) * (ih / h);
  const sw = ((2 * R) / zoom) * (iw / w);
  const sh = ((2 * R) / zoom) * (ih / h);

  ctx.save();
  ctx.beginPath();
  ctx.arc(lx, ly, R, 0, Math.PI * 2);
  ctx.clip();
  ctx.drawImage(img, sx0, sy0, sw, sh, lx - R, ly - R, 2 * R, 2 * R);
  ctx.restore();

  ctx.beginPath();
  ctx.arc(lx, ly, R, 0, Math.PI * 2);
  ctx.strokeStyle = "#f8fafc";
  ctx.lineWidth = 2;
  ctx.stroke();
}

export default function QuadEditor({
  imageUrl,
  initialQuad,
  onChange,
}: QuadEditorProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const onChangeRef = useRef(onChange);
  onChangeRef.current = onChange;

  const [img, setImg] = useState<HTMLImageElement | null>(null);
  const [display, setDisplay] = useState<{ w: number; h: number } | null>(null);
  const [quad, setQuad] = useState<XY[]>(() => toXY(initialQuad));
  const [undoStack, setUndoStack] = useState<XY[][]>([]);
  const [userAdjusted, setUserAdjusted] = useState(false);
  const [dragIndex, setDragIndex] = useState<number | null>(null);
  const dragRef = useRef<{ index: number; start: XY[]; moved: boolean } | null>(
    null
  );

  // Load the preview image.
  useEffect(() => {
    const image = new Image();
    image.src = imageUrl;
    image.onload = () => setImg(image);
    image.onerror = () => setImg(null);
    return () => {
      image.onload = null;
      image.onerror = null;
    };
  }, [imageUrl]);

  // Fit the canvas to the container width while preserving aspect ratio.
  useEffect(() => {
    if (!img) return;
    const container = containerRef.current;
    if (!container) return;
    const update = () => {
      const cw = container.clientWidth || 320;
      const aspect = img.naturalHeight / img.naturalWidth;
      setDisplay({ w: cw, h: Math.max(120, Math.round(cw * aspect)) });
    };
    update();
    const ro = new ResizeObserver(update);
    ro.observe(container);
    return () => ro.disconnect();
  }, [img]);

  // Report the current quad to the parent.
  useEffect(() => {
    onChangeRef.current?.(toPoints(quad), userAdjusted);
  }, [quad, userAdjusted]);

  // Draw.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !img || !display) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(display.w * dpr);
    canvas.height = Math.round(display.h * dpr);
    canvas.style.width = `${display.w}px`;
    canvas.style.height = `${display.h}px`;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, display.w, display.h);
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, display.w, display.h);
    ctx.drawImage(img, 0, 0, display.w, display.h);

    const pts = quad.map((p) => ({ x: p.x * display.w, y: p.y * display.h }));

    // Edges.
    ctx.strokeStyle = "#34d399";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(pts[0].x, pts[0].y);
    for (let i = 1; i < 4; i++) ctx.lineTo(pts[i].x, pts[i].y);
    ctx.closePath();
    ctx.stroke();

    // Handles (numbered TL,TR,BR,BL).
    for (let i = 0; i < 4; i++) {
      const active = dragIndex === i;
      ctx.beginPath();
      ctx.arc(pts[i].x, pts[i].y, active ? 14 : 12, 0, Math.PI * 2);
      ctx.fillStyle = active ? "#fbbf24" : "#34d399";
      ctx.fill();
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.fillStyle = "#0f172a";
      ctx.font = "bold 12px system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(String(i + 1), pts[i].x, pts[i].y + 0.5);
    }

    if (dragIndex !== null) {
      drawLoupe(ctx, img, display.w, display.h, pts[dragIndex], dragIndex);
    }
  }, [img, display, quad, dragIndex]);

  function hitTest(px: number, py: number): number {
    if (!display) return -1;
    let best = -1;
    let bestD = Infinity;
    for (let i = 0; i < 4; i++) {
      const x = quad[i].x * display.w;
      const y = quad[i].y * display.h;
      const d = Math.hypot(px - x, py - y);
      if (d <= HIT_RADIUS && d < bestD) {
        best = i;
        bestD = d;
      }
    }
    return best;
  }

  function onPointerDown(e: React.PointerEvent<HTMLCanvasElement>) {
    if (!display) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const idx = hitTest(e.clientX - rect.left, e.clientY - rect.top);
    if (idx < 0) return;
    e.preventDefault();
    canvas.setPointerCapture(e.pointerId);
    dragRef.current = { index: idx, start: cloneQuad(quad), moved: false };
    setDragIndex(idx);
  }

  function onPointerMove(e: React.PointerEvent<HTMLCanvasElement>) {
    const drag = dragRef.current;
    if (!drag || !display) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const nx = Math.max(0, Math.min(1, (e.clientX - rect.left) / display.w));
    const ny = Math.max(0, Math.min(1, (e.clientY - rect.top) / display.h));

    const next = quad.map((p, i) =>
      i === drag.index ? { x: nx, y: ny } : { ...p }
    );
    if (!isConvex(next)) return; // would cross an edge — revert (keep last)

    if (!drag.moved) {
      setUndoStack((s) => [...s, cloneQuad(drag.start)]);
      drag.moved = true;
      setUserAdjusted(true);
    }
    setQuad(next);
  }

  function onPointerUp(e: React.PointerEvent<HTMLCanvasElement>) {
    const canvas = canvasRef.current;
    if (canvas && dragRef.current) {
      try {
        canvas.releasePointerCapture(e.pointerId);
      } catch {
        /* already released */
      }
    }
    dragRef.current = null;
    setDragIndex(null);
  }

  function applyQuad(next: XY[], recordUndo: boolean, markAdjusted: boolean) {
    if (recordUndo) setUndoStack((s) => [...s, cloneQuad(quad)]);
    setQuad(cloneQuad(next));
    if (markAdjusted) setUserAdjusted(true);
  }

  function resetToAuto() {
    applyQuad(toXY(initialQuad), true, false);
  }

  function resetToFullFrame() {
    applyQuad(FULL_FRAME_INSET, true, true);
  }

  function undo() {
    setUndoStack((stack) => {
      if (stack.length === 0) return stack;
      const prev = stack[stack.length - 1];
      setQuad(cloneQuad(prev));
      return stack.slice(0, -1);
    });
  }

  return (
    <div>
      <div ref={containerRef} className="quad-editor">
        {img ? (
          <canvas
            ref={canvasRef}
            className={dragIndex !== null ? "dragging" : ""}
            onPointerDown={onPointerDown}
            onPointerMove={onPointerMove}
            onPointerUp={onPointerUp}
            onPointerCancel={onPointerUp}
          />
        ) : (
          <div className="loading-block">Loading preview…</div>
        )}
      </div>

      <div className="quad-toolbar">
        <button type="button" className="small ghost" onClick={resetToAuto}>
          Reset to auto
        </button>
        <button type="button" className="small ghost" onClick={resetToFullFrame}>
          Reset to full frame
        </button>
        <button
          type="button"
          className="small ghost"
          onClick={undo}
          disabled={undoStack.length === 0}
        >
          Undo{undoStack.length > 0 ? ` (${undoStack.length})` : ""}
        </button>
        <span className="muted" style={{ fontSize: "0.78rem", alignSelf: "center" }}>
          drag {HANDLE_ORDER.join(" · ")}
        </span>
      </div>
    </div>
  );
}
