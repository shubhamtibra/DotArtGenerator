import { useEffect, useState } from "react";
import { decodeColorMap } from "../utils/colorMap";

/** Decoded per-character RGBA colours for a `color_map`, or null until ready / when unsupported. */
export function useColorMap(encoded, columns, rows) {
  const [decoded, setDecoded] = useState({ encoded: null, pixels: null });

  useEffect(() => {
    if (!encoded) return undefined;
    let cancelled = false;
    decodeColorMap(encoded, columns, rows).then((pixels) => {
      if (!cancelled) setDecoded({ encoded, pixels });
    });
    return () => {
      cancelled = true;
    };
  }, [encoded, columns, rows]);

  return encoded && decoded.encoded === encoded ? decoded.pixels : null;
}
