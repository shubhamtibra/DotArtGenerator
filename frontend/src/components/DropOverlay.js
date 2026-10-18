import { UploadIcon } from "./ui/icons";

/** Full-window affordance shown while a file is dragged over the page. */
export function DropOverlay() {
  return (
    <div className="pointer-events-none fixed inset-0 z-50 flex bg-zinc-950/95 p-4" aria-hidden="true">
      <div className="flex flex-1 flex-col items-center justify-center gap-3 rounded-2xl border-2 border-dashed border-violet-500/70 bg-violet-500/[0.04]">
        <span className="flex h-12 w-12 items-center justify-center rounded-xl bg-violet-500/15 text-violet-300">
          <UploadIcon className="h-6 w-6" />
        </span>
        <p className="text-base font-medium text-zinc-100">Drop to convert</p>
        <p className="text-sm text-zinc-400">PNG, JPEG, GIF, WebP, BMP or TIFF up to 10&nbsp;MB</p>
      </div>
    </div>
  );
}
