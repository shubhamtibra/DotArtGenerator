import { ZOOM_STEPS } from "../constants";
import { Button } from "./ui/Button";
import { SegmentedControl } from "./ui/SegmentedControl";
import {
  CheckIcon,
  CopyIcon,
  DownloadIcon,
  FitWidthIcon,
  MinusIcon,
  MoonIcon,
  PaletteIcon,
  PlusIcon,
  SunIcon,
} from "./ui/icons";

const BACKGROUND_OPTIONS = [
  { value: "dark", label: "Dark background", icon: MoonIcon },
  { value: "light", label: "Light background", icon: SunIcon },
];

const MIN_ZOOM = ZOOM_STEPS[0];
const MAX_ZOOM = ZOOM_STEPS[ZOOM_STEPS.length - 1];

function nextZoomStep(fontSize, direction) {
  if (direction > 0) return ZOOM_STEPS.find((step) => step > fontSize + 0.01) ?? MAX_ZOOM;
  return [...ZOOM_STEPS].reverse().find((step) => step < fontSize - 0.01) ?? MIN_ZOOM;
}

/** Preview background, colour and zoom on the left; copy and export on the right. */
export function ViewerToolbar({ view, onViewChange, fontSize, hasArt, copied, onCopy, onDownloadText, onDownloadPng }) {
  const zoomTo = (zoom) => onViewChange({ zoom });
  return (
    <div className="flex flex-wrap items-center justify-between gap-2 border-b border-zinc-800 px-2.5 py-2">
      <div className="flex flex-wrap items-center gap-1.5">
        <SegmentedControl
          label="Preview background"
          size="sm"
          iconOnly
          value={view.background}
          options={BACKGROUND_OPTIONS}
          onChange={(background) => onViewChange({ background })}
        />
        <Button
          size="sm"
          icon={PaletteIcon}
          aria-pressed={view.color}
          onClick={() => onViewChange({ color: !view.color })}
        >
          Color
        </Button>
        <span className="mx-1 hidden h-4 w-px bg-zinc-800 sm:block" aria-hidden="true" />
        <div className="flex items-center gap-0.5" role="group" aria-label="Zoom">
          <Button
            variant="ghost"
            size="sm"
            iconOnly
            icon={MinusIcon}
            aria-label="Zoom out"
            title="Zoom out"
            disabled={!hasArt || fontSize <= MIN_ZOOM}
            onClick={() => zoomTo(nextZoomStep(fontSize, -1))}
          />
          <Button
            variant="ghost"
            size="sm"
            icon={FitWidthIcon}
            aria-pressed={view.zoom === "fit"}
            title="Fit to view"
            className="tabular-nums"
            onClick={() => zoomTo("fit")}
          >
            {view.zoom === "fit" ? "Fit" : `${fontSize}px`}
          </Button>
          <Button
            variant="ghost"
            size="sm"
            iconOnly
            icon={PlusIcon}
            aria-label="Zoom in"
            title="Zoom in"
            disabled={!hasArt || fontSize >= MAX_ZOOM}
            onClick={() => zoomTo(nextZoomStep(fontSize, 1))}
          />
        </div>
      </div>

      <div className="flex items-center gap-1.5">
        <Button size="sm" icon={copied ? CheckIcon : CopyIcon} disabled={!hasArt} onClick={onCopy}>
          Copy
        </Button>
        <Button
          size="sm"
          icon={DownloadIcon}
          aria-label="Download TXT"
          title="Download as .txt"
          disabled={!hasArt}
          onClick={onDownloadText}
        >
          TXT
        </Button>
        <Button
          size="sm"
          icon={DownloadIcon}
          aria-label="Download PNG"
          title="Download as .png"
          disabled={!hasArt}
          onClick={onDownloadPng}
        >
          PNG
        </Button>
      </div>
    </div>
  );
}
