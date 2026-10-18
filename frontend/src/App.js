import { useCallback, useEffect, useRef, useState } from "react";
import { API_URL } from "./config";
import { DEFAULT_SETTINGS, SAMPLES, STORAGE_KEYS } from "./constants";
import { ArtViewer } from "./components/ArtViewer";
import { ControlsPanel } from "./components/ControlsPanel";
import { DropOverlay } from "./components/DropOverlay";
import { EmptyState } from "./components/EmptyState";
import { ErrorBanner } from "./components/ErrorBanner";
import { Header } from "./components/Header";
import { SourceCard } from "./components/SourceCard";
import { useClipboardPaste } from "./hooks/useClipboardPaste";
import { useDotArt } from "./hooks/useDotArt";
import { useImageSource } from "./hooks/useImageSource";
import { useStoredState } from "./hooks/useStoredState";
import { useWindowFileDrop } from "./hooks/useWindowFileDrop";
import { cx } from "./utils/cx";
import {
  applyPreset,
  buildRequestOptions,
  estimateGrid,
  getSettingsIssue,
  sanitizeSettings,
  sanitizeView,
} from "./utils/settings";

function generationErrorHint(error) {
  if (error?.code !== "network_error") return null;
  const code = "text-zinc-300";
  return (
    <>
      Start it with <code className={code}>python manage.py runserver</code> in <code className={code}>backend/</code>,
      or point <code className={code}>REACT_APP_API_URL</code> at your server (currently{" "}
      <code className={code}>{API_URL}</code>).
    </>
  );
}

function announcementFor(isPreparing, art) {
  if (isPreparing) return "Preparing image…";
  if (art.status === "loading") return "Generating dot art…";
  if (art.status === "success" && art.result) {
    return `Dot art ready: ${art.result.columns} by ${art.result.rows} characters.`;
  }
  return "";
}

export default function App() {
  const [settings, setSettings] = useStoredState(STORAGE_KEYS.settings, sanitizeSettings);
  const [view, setView] = useStoredState(STORAGE_KEYS.view, sanitizeView);
  const [sourcePreview, setSourcePreview] = useState("original");
  const fileInputRef = useRef(null);

  const { source, isLoading: isPreparing, error: sourceError, loadFile, loadSample, loadUrl, clear, dismissError } =
    useImageSource();

  const issue = getSettingsIssue(settings);
  const requestOptions = buildRequestOptions(settings, {
    background: view.background,
    color: view.color,
    includeProcessed: sourcePreview === "processed",
  });
  const art = useDotArt(source, requestOptions, { enabled: !issue });

  const openFilePicker = useCallback(() => fileInputRef.current?.click(), []);
  const updateSettings = useCallback((patch) => setSettings((current) => ({ ...current, ...patch })), [setSettings]);
  const updateView = useCallback((patch) => setView((current) => ({ ...current, ...patch })), [setView]);
  const loadPasted = useCallback((file) => loadFile(file, { kind: "paste", name: "Pasted image" }), [loadFile]);

  const isDragging = useWindowFileDrop(loadFile);
  useClipboardPaste({ onFile: loadPasted, onUrl: loadUrl });

  // ?sample=<id> loads a bundled sample on start (handy for demos and screenshots).
  useEffect(() => {
    const requested = new URLSearchParams(window.location.search).get("sample");
    const sample = SAMPLES.find((candidate) => candidate.id === requested);
    if (sample) loadSample(sample);
  }, [loadSample]);

  const handleFileInput = (event) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (file) loadFile(file);
  };

  const knownSize =
    source?.width && source?.height
      ? { width: source.width, height: source.height }
      : art.result?.source
        ? { width: art.result.source.width, height: art.result.source.height }
        : null;
  const grid = estimateGrid(settings, knownSize ? knownSize.height / knownSize.width : null);

  return (
    <div className="flex min-h-screen flex-col bg-zinc-950 text-zinc-100">
      <Header hasSource={Boolean(source)} onNewImage={openFilePicker} />
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        aria-label="Upload image"
        tabIndex={-1}
        className="hidden"
        onChange={handleFileInput}
      />

      {source ? (
        <div className="flex flex-1 flex-col lg:h-[calc(100dvh-3rem)] lg:flex-none lg:flex-row lg:overflow-hidden">
          <aside
            aria-label="Image and settings"
            className={cx(
              "order-2 border-t border-zinc-800/80 bg-zinc-900/30",
              "lg:order-1 lg:w-[340px] lg:shrink-0 lg:overflow-y-auto lg:border-r lg:border-t-0"
            )}
          >
            <SourceCard
              source={source}
              dimensions={knownSize}
              preview={sourcePreview}
              onPreviewChange={setSourcePreview}
              processedImage={art.result?.processed_image}
              isProcessing={art.isLoading}
              onReplace={openFilePicker}
              onRemove={clear}
            />
            <div className="border-t border-zinc-800/80">
              <ControlsPanel
                settings={settings}
                onChange={updateSettings}
                onApplyPreset={(preset) => setSettings((current) => applyPreset(current, preset))}
                onReset={() => setSettings({ ...DEFAULT_SETTINGS })}
                grid={grid}
                resolvedThreshold={art.result?.options?.threshold}
                issue={issue}
              />
            </div>
          </aside>

          <main className="order-1 flex min-w-0 flex-1 flex-col gap-3 p-3 sm:p-4 lg:order-2 lg:p-5">
            {sourceError && (
              <ErrorBanner title="Couldn't use that image" message={sourceError} onDismiss={dismissError} />
            )}
            {art.error && (
              <ErrorBanner
                title="Couldn't generate dot art"
                message={art.error.message}
                hint={generationErrorHint(art.error)}
                onRetry={art.retry}
                onDismiss={art.dismissError}
              />
            )}
            <ArtViewer
              result={art.result}
              status={art.status}
              view={view}
              onViewChange={updateView}
              sourceName={source.name}
              placeholder={issue}
            />
          </main>
        </div>
      ) : (
        <EmptyState
          isLoading={isPreparing}
          error={sourceError}
          onDismissError={dismissError}
          onBrowse={openFilePicker}
          onUrl={loadUrl}
          onSample={loadSample}
        />
      )}

      {isDragging && <DropOverlay />}
      <div className="sr-only" role="status" aria-live="polite">
        {announcementFor(isPreparing, art)}
      </div>
    </div>
  );
}
