import { SAMPLES } from "../constants";
import { sampleThumbnailUrl } from "../hooks/useImageSource";
import { cx } from "../utils/cx";
import { focusRing } from "./ui/styles";

export function SampleGallery({ onSelect, disabled = false }) {
  return (
    <section aria-labelledby="samples-heading">
      <h2 id="samples-heading" className="text-xs font-medium text-zinc-500">
        No image handy? Start from a sample
      </h2>
      <ul className="mt-3 grid grid-cols-3 gap-2.5 sm:gap-3">
        {SAMPLES.map((sample) => (
          <li key={sample.id}>
            <button
              type="button"
              disabled={disabled}
              onClick={() => onSelect(sample)}
              className={cx(
                "group block w-full overflow-hidden rounded-lg border border-zinc-800 bg-zinc-900 text-left transition-colors",
                "hover:border-zinc-600 disabled:cursor-wait disabled:opacity-60",
                focusRing
              )}
            >
              <span className="block aspect-[4/3] overflow-hidden bg-zinc-950">
                <img
                  src={sampleThumbnailUrl(sample)}
                  alt=""
                  loading="lazy"
                  className="h-full w-full object-cover opacity-90 transition duration-300 group-hover:scale-[1.03] group-hover:opacity-100"
                />
              </span>
              <span className="block border-t border-zinc-800 px-2.5 py-2 sm:px-3">
                <span className="block truncate text-[13px] font-medium text-zinc-200">{sample.label}</span>
                <span className="mt-0.5 hidden truncate text-xs text-zinc-500 sm:block">{sample.description}</span>
              </span>
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}
