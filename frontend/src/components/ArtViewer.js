import { useEffect, useMemo, useRef, useState } from "react";
import { ART_FONT_FAMILY, FIT_FONT_SIZE, MODE_GLYPHS, MODES, PREVIEW_THEMES } from "../constants";
import { useColorMap } from "../hooks/useColorMap";
import { useElementSize } from "../hooks/useElementSize";
import { copyText } from "../utils/clipboard";
import { cx } from "../utils/cx";
import { canDrawOnCanvas, cellHeightFor } from "../utils/drawArt";
import { artFilename, downloadBlob, downloadText, renderArtPng } from "../utils/export";
import { measureAdvance } from "../utils/fontMetrics";
import { clamp, formatNumber } from "../utils/format";
import { ArtCanvas } from "./ArtCanvas";
import { ViewerToolbar } from "./ViewerToolbar";
import { Spinner } from "./ui/Spinner";
import { ImageIcon } from "./ui/icons";

const STAGE_PADDING = 24;
const NOTICE_MS = 2500;

const MODE_LABELS = Object.fromEntries(MODES.map((mode) => [mode.value, mode.label]));

/** Font size at which the whole grid fits the stage. */
function fitFontSize(stage, { mode, columns, rows, charAspect, advance }) {
  if (!columns || !stage.width || !stage.height) return 8;
  const cellWidth = Math.min(
    (stage.width - STAGE_PADDING * 2) / columns,
    (stage.height - STAGE_PADDING * 2) / (rows * cellHeightFor(mode, 1, charAspect))
  );
  return clamp(cellWidth / advance, FIT_FONT_SIZE.min, FIT_FONT_SIZE.max);
}

/** Transient success/error message for copy and export actions. */
function useNotice() {
  const [notice, setNotice] = useState(null);
  useEffect(() => {
    if (!notice) return undefined;
    const timer = setTimeout(() => setNotice(null), NOTICE_MS);
    return () => clearTimeout(timer);
  }, [notice]);
  return [notice, setNotice];
}

/**
 * The art preview: toolbar (background, colour, zoom, copy/export), the
 * scrollable stage, and a status bar with output stats.
 */
export function ArtViewer({ result, status, view, onViewChange, sourceName, placeholder }) {
  const stageRef = useRef(null);
  const stage = useElementSize(stageRef);
  const [notice, setNotice] = useNotice();

  const theme = PREVIEW_THEMES[view.background];
  const isLoading = status === "loading";
  const mode = result?.mode ?? "braille";
  const columns = result?.columns ?? 0;
  const rows = result?.rows ?? 0;
  const charAspect = result?.options?.char_aspect ?? 0.5;
  const advance = useMemo(() => measureAdvance(MODE_GLYPHS[mode] ?? "@"), [mode]);

  const fontSize = view.zoom === "fit" ? fitFontSize(stage, { mode, columns, rows, charAspect, advance }) : view.zoom;
  const cellWidth = fontSize * advance;
  const cellHeight = cellHeightFor(mode, cellWidth, charAspect);
  const colors = useColorMap(view.color ? result?.color_map : null, columns, rows);
  // Braille is drawn as real dots: fallback Braille fonts (e.g. on macOS) render
  // small, unevenly spaced dots. Colour always needs the canvas. A transparent
  // <pre> on top keeps the text selectable either way.
  const drawsArt = canDrawOnCanvas() && (mode === "braille" || Boolean(colors));

  const handleCopy = async () => {
    const copied = await copyText(result.art);
    setNotice(
      copied
        ? { action: "copy", tone: "success", text: "Copied to clipboard" }
        : { action: "copy", tone: "error", text: "Couldn't copy. Select the art and copy it manually." }
    );
  };

  const handleDownloadText = () => {
    const filename = artFilename(sourceName, result, "txt");
    downloadText(result.art, filename);
    setNotice({ tone: "success", text: `Saved ${filename}` });
  };

  const handleDownloadPng = async () => {
    const blob = await renderArtPng({ art: result.art, mode, columns, rows, charAspect, colors, theme });
    if (!blob) {
      setNotice({ tone: "error", text: "PNG export isn't supported in this browser." });
      return;
    }
    const filename = artFilename(sourceName, result, "png");
    downloadBlob(blob, filename);
    setNotice({ tone: "success", text: `Saved ${filename}` });
  };

  return (
    <section
      aria-label="Dot art preview"
      className="flex min-h-0 flex-1 flex-col overflow-hidden rounded-xl border border-zinc-800 bg-zinc-900 shadow-sm"
    >
      <ViewerToolbar
        view={view}
        onViewChange={onViewChange}
        fontSize={fontSize}
        hasArt={Boolean(result)}
        copied={notice?.action === "copy" && notice.tone === "success"}
        onCopy={handleCopy}
        onDownloadText={handleDownloadText}
        onDownloadPng={handleDownloadPng}
      />

      <div
        className="relative h-[62vh] min-h-[300px] lg:h-auto lg:min-h-0 lg:flex-1"
        style={{ background: theme.background }}
      >
        <div
          ref={stageRef}
          role="region"
          aria-label="Art canvas"
          tabIndex={result ? 0 : -1}
          className="absolute inset-0 overflow-auto"
        >
          {result && (
            <div className="flex min-h-full w-max min-w-full" style={{ padding: STAGE_PADDING }}>
              <div className={cx("relative m-auto transition-opacity duration-150", isLoading && "opacity-40")}>
                {drawsArt && (
                  <ArtCanvas
                    art={result.art}
                    mode={mode}
                    columns={columns}
                    rows={rows}
                    cellWidth={cellWidth}
                    cellHeight={cellHeight}
                    colors={colors}
                    foreground={theme.foreground}
                  />
                )}
                <pre
                  className={cx("m-0 select-text", drawsArt && "absolute inset-0 selection:bg-violet-500/40")}
                  style={{
                    fontFamily: ART_FONT_FAMILY,
                    fontSize: `${fontSize}px`,
                    lineHeight: `${cellHeight}px`,
                    letterSpacing: 0,
                    color: drawsArt ? "transparent" : theme.foreground,
                  }}
                >
                  {result.art}
                </pre>
              </div>
            </div>
          )}
        </div>

        {result && isLoading && (
          <div className="pointer-events-none absolute right-3 top-3 flex items-center gap-2 rounded-full border border-zinc-700/80 bg-zinc-900/95 px-2.5 py-1 text-xs text-zinc-300 shadow-lg">
            <Spinner className="h-3 w-3" />
            Rendering
          </div>
        )}
        {!result && (
          <StagePlaceholder
            isLoading={isLoading}
            status={status}
            placeholder={placeholder}
            dark={view.background === "dark"}
          />
        )}
      </div>

      <StatusBar result={result} status={status} notice={notice} />
    </section>
  );
}

function StagePlaceholder({ isLoading, status, placeholder, dark }) {
  let title;
  let body;
  if (isLoading) {
    title = "Generating preview…";
  } else if (placeholder) {
    title = "Nothing to render yet";
    body = placeholder;
  } else if (status === "error") {
    title = "No preview yet";
    body = "Your art will appear here once the generator responds.";
  } else {
    return null;
  }
  return (
    <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 p-6 text-center">
      <span
        className={cx(
          "flex h-10 w-10 items-center justify-center rounded-xl border",
          dark ? "border-zinc-800 bg-zinc-900 text-zinc-400" : "border-zinc-200 bg-zinc-50 text-zinc-500"
        )}
      >
        {isLoading ? <Spinner className="h-4 w-4" /> : <ImageIcon className="h-4 w-4" />}
      </span>
      <div>
        <p className={cx("text-sm font-medium", dark ? "text-zinc-300" : "text-zinc-700")}>{title}</p>
        {body && <p className="mx-auto mt-1 max-w-xs text-xs leading-relaxed text-zinc-500">{body}</p>}
      </div>
    </div>
  );
}

function StatusBar({ result, status, notice }) {
  const stats = result
    ? [
        ["Mode", MODE_LABELS[result.mode] ?? result.mode],
        ["Size", `${formatNumber(result.columns)} × ${formatNumber(result.rows)}`],
        ["Characters", `${formatNumber(result.art.length)} chars`],
        ...(Number.isFinite(result.elapsed_ms) ? [["Render time", `${formatNumber(result.elapsed_ms)} ms`]] : []),
      ]
    : [];

  return (
    <div className="flex min-h-[2.25rem] flex-wrap items-center justify-between gap-x-4 gap-y-1 border-t border-zinc-800 px-3 py-2 text-xs text-zinc-500">
      {stats.length > 0 ? (
        <dl className="flex flex-wrap items-center gap-x-2 gap-y-1 tabular-nums">
          {stats.map(([label, value], index) => (
            <div key={label} className="flex items-center gap-2">
              {index > 0 && (
                <span aria-hidden="true" className="text-zinc-700">
                  ·
                </span>
              )}
              <dt className="sr-only">{label}</dt>
              <dd className={index === 1 ? "text-zinc-300" : undefined}>{value}</dd>
            </div>
          ))}
        </dl>
      ) : (
        <span>No output yet</span>
      )}

      <div className="flex items-center gap-1.5">
        <span className="sr-only" aria-live="polite">
          {notice?.text ?? ""}
        </span>
        {notice ? (
          <span className={notice.tone === "error" ? "text-red-400" : "text-emerald-400"}>{notice.text}</span>
        ) : (
          <GenerationStatus status={status} />
        )}
      </div>
    </div>
  );
}

const STATUS_STYLES = {
  loading: { dot: "bg-violet-400 animate-pulse", label: "Rendering…" },
  success: { dot: "bg-emerald-400", label: "Live preview" },
  error: { dot: "bg-red-400", label: "Render failed" },
  idle: { dot: "bg-zinc-600", label: "Idle" },
};

function GenerationStatus({ status }) {
  const { dot, label } = STATUS_STYLES[status] ?? STATUS_STYLES.idle;
  return (
    <span className="inline-flex items-center gap-1.5">
      <span aria-hidden="true" className={cx("h-1.5 w-1.5 rounded-full", dot)} />
      {label}
    </span>
  );
}
