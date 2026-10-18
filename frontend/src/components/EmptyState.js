import { ART_FONT_FAMILY } from "../constants";
import { cx } from "../utils/cx";
import { BrailleGlyphs } from "./BrailleGlyphs";
import { ErrorBanner } from "./ErrorBanner";
import { SampleGallery } from "./SampleGallery";
import { UrlForm } from "./UrlForm";
import { Kbd } from "./ui/Kbd";
import { Spinner } from "./ui/Spinner";
import { UploadIcon } from "./ui/icons";
import { focusRing } from "./ui/styles";

const IS_MAC = typeof navigator !== "undefined" && /Mac|iPhone|iPad/.test(navigator.userAgent);

const OUTPUT_KINDS = [
  { label: "Braille", sample: <BrailleGlyphs text="⣿⣷⡄" className="h-3" /> },
  { label: "ASCII", sample: "@%#" },
  { label: "Blocks", sample: "█▓░" },
];

/** First-run screen: choose an image by dropping, browsing, pasting, URL or sample. */
export function EmptyState({ isLoading, error, onDismissError, onBrowse, onUrl, onSample }) {
  return (
    <main className="mx-auto w-full max-w-2xl flex-1 px-4 pb-16 pt-10 sm:pt-16 lg:pt-20">
      <div className="text-center">
        <ul className="inline-flex flex-wrap items-center justify-center gap-1.5" aria-label="Output styles">
          {OUTPUT_KINDS.map(({ sample, label }) => (
            <li
              key={label}
              className={cx(
                "inline-flex h-6 items-center gap-1.5 rounded-full border border-zinc-800 bg-zinc-900/60",
                "pl-2 pr-2.5 text-xs text-zinc-400"
              )}
            >
              <span
                aria-hidden="true"
                className="text-xs leading-none text-violet-300"
                style={{ fontFamily: ART_FONT_FAMILY }}
              >
                {sample}
              </span>
              {label}
            </li>
          ))}
        </ul>
        <h1 className="mt-5 text-balance text-[1.75rem] font-semibold leading-tight tracking-tight text-zinc-50 sm:text-4xl">
          Turn any image into dot art
        </h1>
        <p className="mx-auto mt-3 max-w-md text-pretty text-[15px] leading-relaxed text-zinc-400">
          Text art that pastes anywhere — chats, READMEs, terminals. Tune it live, then copy it or export a PNG.
        </p>
      </div>

      {error && (
        <ErrorBanner className="mt-8" title="Couldn't use that image" message={error} onDismiss={onDismissError} />
      )}

      <button
        type="button"
        onClick={onBrowse}
        disabled={isLoading}
        className={cx(
          "group mt-8 flex w-full flex-col items-center rounded-xl border border-dashed border-zinc-700 bg-zinc-900/40",
          "px-6 py-10 text-center transition-colors hover:border-zinc-500 hover:bg-zinc-900/80 disabled:cursor-wait sm:py-12",
          focusRing
        )}
      >
        <span
          className={cx(
            "flex h-11 w-11 items-center justify-center rounded-xl border border-zinc-800 bg-zinc-900",
            "text-zinc-300 shadow-sm transition-colors group-hover:text-violet-300"
          )}
        >
          {isLoading ? <Spinner className="h-5 w-5" /> : <UploadIcon className="h-5 w-5" />}
        </span>
        <span className="mt-4 text-sm font-medium text-zinc-100">
          {isLoading ? (
            "Preparing image…"
          ) : (
            <>
              <span className="sm:hidden">
                Tap to <span className="text-violet-400">choose an image</span>
              </span>
              <span className="hidden sm:inline">
                Drop an image here or <span className="text-violet-400 group-hover:text-violet-300">browse</span>
              </span>
            </>
          )}
        </span>
        <span className="mt-1.5 text-balance text-xs text-zinc-500">
          PNG, JPEG, GIF, WebP, BMP or TIFF · up to 10&nbsp;MB
        </span>
        <span className="mt-4 hidden items-center gap-1 text-xs text-zinc-500 sm:inline-flex">
          You can also paste an image with <Kbd>{IS_MAC ? "⌘" : "Ctrl"}</Kbd>
          <Kbd>V</Kbd>
        </span>
      </button>

      <div className="mt-3">
        <UrlForm onSubmit={onUrl} disabled={isLoading} />
      </div>

      <div className="mt-10">
        <SampleGallery onSelect={onSample} disabled={isLoading} />
      </div>
    </main>
  );
}
