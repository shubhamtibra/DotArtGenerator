import { useEffect, useState } from "react";

const EMPTY = { width: 0, height: 0 };

/** Client size of the element in `ref` (padding included, scrollbars excluded), kept current. */
export function useElementSize(ref) {
  const [size, setSize] = useState(EMPTY);
  useEffect(() => {
    const element = ref.current;
    if (!element) return undefined;
    const measure = () => setSize({ width: element.clientWidth, height: element.clientHeight });
    measure();
    if (typeof ResizeObserver === "undefined") return undefined;
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    return () => observer.disconnect();
  }, [ref]);
  return size;
}
