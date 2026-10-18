import { useId, useState } from "react";
import { cx } from "../utils/cx";
import { validateImageUrl } from "../utils/image";
import { Button } from "./ui/Button";
import { LinkIcon } from "./ui/icons";
import { focusRing } from "./ui/styles";

/** Load an image by URL; validates http(s) before handing it on. */
export function UrlForm({ onSubmit, disabled = false }) {
  const [value, setValue] = useState("");
  const [error, setError] = useState(null);
  const errorId = useId();

  const handleSubmit = (event) => {
    event.preventDefault();
    const { url, error: problem } = validateImageUrl(value);
    if (problem) {
      setError(problem);
      return;
    }
    setError(null);
    onSubmit(url);
  };

  return (
    <form onSubmit={handleSubmit} noValidate>
      <div className="flex gap-2">
        <div className="relative min-w-0 flex-1">
          <LinkIcon className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-zinc-500" />
          <input
            type="url"
            inputMode="url"
            autoComplete="off"
            spellCheck="false"
            aria-label="Image URL"
            aria-invalid={Boolean(error)}
            aria-describedby={error ? errorId : undefined}
            placeholder="Paste an image URL"
            value={value}
            disabled={disabled}
            onChange={(event) => {
              setValue(event.target.value);
              if (error) setError(null);
            }}
            className={cx(
              "h-10 w-full rounded-lg border bg-zinc-900/60 pl-9 pr-3 text-sm text-zinc-100 shadow-sm placeholder:text-zinc-500",
              "transition-colors hover:border-zinc-700 focus:border-violet-500/60",
              error ? "border-red-500/50" : "border-zinc-800",
              focusRing
            )}
          />
        </div>
        <Button type="submit" size="lg" disabled={disabled}>
          Load
        </Button>
      </div>
      {error && (
        <p id={errorId} className="mt-2 text-xs text-red-400">
          {error}
        </p>
      )}
    </form>
  );
}
