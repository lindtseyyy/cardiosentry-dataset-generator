import type { ModelInputSize } from "../api/types";

/** "768\u00d71024", or "768px" when the model input is square. */
export function formatInputSize(size?: ModelInputSize | null): string | null {
  if (!Array.isArray(size) || size.length !== 2) return null;
  const [h, w] = size;
  return h === w ? `${h}px` : `${h}\u00d7${w}`;
}

export interface ModelBadgeProps {
  id?: string;
  displayName: string;
  version?: string;
  checkpointShort?: string;
  inputSize?: ModelInputSize | null;
  device?: string;
}

export default function ModelBadge({
  id,
  displayName,
  version,
  checkpointShort,
  inputSize,
  device,
}: ModelBadgeProps) {
  return (
    <div className="model-badge">
      <span className="name">{displayName}</span>
      {version ? <span className="meta">v{version}</span> : null}
      {id ? <span className="meta mono">{id}</span> : null}
      {checkpointShort ? (
        <span className="mono">sha&nbsp;{checkpointShort}</span>
      ) : null}
      {formatInputSize(inputSize) ? (
        <span className="meta">input&nbsp;{formatInputSize(inputSize)}</span>
      ) : null}
      {device ? <span className="meta">{device}</span> : null}
    </div>
  );
}
