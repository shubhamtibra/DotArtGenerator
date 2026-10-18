import { cx } from "../utils/cx";
import { Button } from "./ui/Button";
import { AlertIcon, CloseIcon, RetryIcon } from "./ui/icons";
import { focusRing } from "./ui/styles";

/** Dismissible error callout with an optional Retry action. */
export function ErrorBanner({ title, message, hint, onRetry, onDismiss, className }) {
  return (
    <div
      role="alert"
      className={cx(
        "flex items-start gap-3 rounded-xl border border-red-500/25 bg-red-500/[0.07] p-3 pr-2 sm:p-4 sm:pr-3",
        className
      )}
    >
      <AlertIcon className="mt-0.5 h-4 w-4 shrink-0 text-red-400" />
      <div className="min-w-0 flex-1">
        {title && <p className="text-[13px] font-medium text-red-200">{title}</p>}
        <p className={cx("text-[13px] leading-relaxed text-red-200/80", title && "mt-0.5")}>{message}</p>
        {hint && <p className="mt-1.5 text-xs leading-relaxed text-zinc-400">{hint}</p>}
        {onRetry && (
          <Button size="sm" icon={RetryIcon} onClick={onRetry} className="mt-3">
            Retry
          </Button>
        )}
      </div>
      {onDismiss && (
        <button
          type="button"
          aria-label="Dismiss error"
          title="Dismiss"
          onClick={onDismiss}
          className={cx(
            "-mt-1 inline-flex h-7 w-7 shrink-0 items-center justify-center rounded-md text-red-300/70 transition-colors hover:bg-red-500/10 hover:text-red-200 sm:-mt-1.5",
            focusRing
          )}
        >
          <CloseIcon className="h-3.5 w-3.5" />
        </button>
      )}
    </div>
  );
}
