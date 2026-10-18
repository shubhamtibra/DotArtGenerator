const numberFormat = new Intl.NumberFormat("en-US");

export function formatNumber(value) {
  return numberFormat.format(value);
}

export function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes < 0) return "";
  if (bytes < 1024) return `${bytes} B`;
  const kb = bytes / 1024;
  if (kb < 1024) return `${Math.round(kb)} KB`;
  const mb = kb / 1024;
  return `${mb >= 10 ? Math.round(mb) : mb.toFixed(1)} MB`;
}

/** "+12", "0", "-5": for sliders centred on zero. */
export function formatSigned(value) {
  return value > 0 ? `+${value}` : String(value);
}

export function formatDecimal(value, digits = 2) {
  return Number(value).toFixed(digits);
}

export function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}
