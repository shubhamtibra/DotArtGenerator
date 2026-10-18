import { forwardRef } from "react";
import { cx } from "../../utils/cx";
import { focusRing } from "./styles";

/** Toggle buttons (aria-pressed) get a "pressed" look per variant. */
const VARIANTS = {
  primary: "bg-violet-600 text-white shadow-sm hover:bg-violet-500 active:bg-violet-700",
  secondary:
    "border border-zinc-800 bg-zinc-900 text-zinc-200 shadow-sm hover:border-zinc-700 hover:bg-zinc-800 " +
    "aria-pressed:border-violet-500/40 aria-pressed:bg-violet-500/15 aria-pressed:text-violet-100 aria-pressed:hover:bg-violet-500/20",
  ghost: "text-zinc-400 hover:bg-zinc-800/80 hover:text-zinc-100 aria-pressed:bg-zinc-800 aria-pressed:text-zinc-100",
};

const SIZES = {
  sm: "h-7 gap-1.5 rounded-md px-2.5 text-xs",
  md: "h-8 gap-2 rounded-md px-3 text-[13px]",
  lg: "h-10 gap-2 rounded-lg px-4 text-sm",
};

const ICON_ONLY_SIZES = {
  sm: "h-7 w-7 rounded-md",
  md: "h-8 w-8 rounded-md",
  lg: "h-10 w-10 rounded-lg",
};

/**
 * Button with an optional leading icon. Icon-only buttons must pass an
 * `aria-label`.
 */
export const Button = forwardRef(function Button(
  { variant = "secondary", size = "md", icon: IconComponent, iconOnly = false, className, children, type = "button", ...props },
  ref
) {
  return (
    <button
      ref={ref}
      type={type}
      className={cx(
        "inline-flex shrink-0 select-none items-center justify-center whitespace-nowrap font-medium transition-colors",
        "disabled:pointer-events-none disabled:opacity-50",
        focusRing,
        VARIANTS[variant],
        iconOnly ? ICON_ONLY_SIZES[size] : SIZES[size],
        className
      )}
      {...props}
    >
      {IconComponent && <IconComponent className={size === "sm" ? "h-3.5 w-3.5" : "h-4 w-4"} />}
      {children}
    </button>
  );
});
