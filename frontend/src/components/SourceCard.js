import { useState } from "react";
import { formatBytes } from "../utils/format";
import { Button } from "./ui/Button";
import { SegmentedControl } from "./ui/SegmentedControl";
import { Spinner } from "./ui/Spinner";
import { ImageIcon, ImagePlusIcon, TrashIcon } from "./ui/icons";

const PREVIEW_OPTIONS = [
  { value: "original", label: "Original" },
  { value: "processed", label: "Processed" },
];

function describeSource(source, dimensions) {
  const parts = [];
  if (source.kind === "sample") parts.push("Sample");
  if (dimensions) parts.push(`${dimensions.width} × ${dimensions.height}`);
  const format = source.type?.replace(/^image\//, "").toUpperCase();
  if (format) parts.push(format === "JPEG" ? "JPG" : format);
  if (source.size) parts.push(formatBytes(source.size));
  return parts.join(" · ");
}

/** Thumbnail that falls back to a placeholder when the browser can't display the image. */
function Thumbnail({ src, alt, pixelated }) {
  const [broken, setBroken] = useState(false);
  if (broken) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-2 text-zinc-500">
        <ImageIcon className="h-5 w-5" />
        <span className="text-xs">Preview not available</span>
      </div>
    );
  }
  return (
    <img
      src={src}
      alt={alt}
      referrerPolicy="no-referrer"
      onError={() => setBroken(true)}
      className="h-full w-full object-contain"
      style={pixelated ? { imageRendering: "pixelated" } : undefined}
    />
  );
}

/**
 * The current source: preview (original, or the server's processed
 * grayscale), name, size and Replace/Remove actions.
 */
export function SourceCard({
  source,
  dimensions,
  preview,
  onPreviewChange,
  processedImage,
  isProcessing,
  onReplace,
  onRemove,
}) {
  const showProcessed = preview === "processed";
  const processedSrc = processedImage ? `data:image/png;base64,${processedImage}` : null;

  return (
    <section aria-label="Source image" className="px-4 pb-4 pt-4">
      <div className="mb-3 flex h-7 items-center justify-between gap-2">
        <h2 className="text-[11px] font-semibold uppercase tracking-[0.08em] text-zinc-500">Source</h2>
        <SegmentedControl
          label="Source preview"
          size="sm"
          value={preview}
          options={PREVIEW_OPTIONS}
          onChange={onPreviewChange}
        />
      </div>

      <div className="relative aspect-[16/10] overflow-hidden rounded-lg border border-zinc-800 bg-zinc-950">
        {showProcessed && processedSrc ? (
          <Thumbnail key={processedSrc} src={processedSrc} alt="Processed grayscale input" pixelated />
        ) : (
          <Thumbnail key={source.previewUrl} src={source.previewUrl} alt={`Preview of ${source.name}`} />
        )}
        {showProcessed && !processedSrc && (
          <div className="absolute inset-0 flex items-center justify-center gap-2 bg-zinc-950/75 text-xs text-zinc-300">
            {isProcessing ? (
              <>
                <Spinner className="h-3.5 w-3.5" />
                Processing…
              </>
            ) : (
              "Available once the art renders"
            )}
          </div>
        )}
      </div>

      <div className="mt-3 flex items-center justify-between gap-3">
        <div className="min-w-0">
          <p className="truncate text-[13px] font-medium text-zinc-100" title={source.detail || source.name}>
            {source.name}
          </p>
          <p className="mt-0.5 truncate text-xs tabular-nums text-zinc-500">
            {describeSource(source, dimensions) || "Remote image"}
          </p>
        </div>
        <div className="flex shrink-0 items-center gap-0.5">
          <Button
            variant="ghost"
            size="sm"
            iconOnly
            icon={ImagePlusIcon}
            aria-label="Replace image"
            title="Replace image"
            onClick={onReplace}
          />
          <Button
            variant="ghost"
            size="sm"
            iconOnly
            icon={TrashIcon}
            aria-label="Remove image"
            title="Remove image"
            onClick={onRemove}
          />
        </div>
      </div>
    </section>
  );
}
