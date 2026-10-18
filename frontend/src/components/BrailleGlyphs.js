import { brailleDots } from "../utils/braille";

const PITCH = 3;

/** Braille text drawn as SVG dots, independent of the fonts installed. */
export function BrailleGlyphs({ text, className = "h-3" }) {
  const chars = Array.from(text);
  const dots = chars.flatMap((char, index) => brailleDots(char).map(([x, y]) => [index * 2 + x, y]));
  return (
    <svg viewBox={`0 0 ${chars.length * 2 * PITCH} ${4 * PITCH}`} aria-hidden="true" className={className}>
      {dots.map(([x, y]) => (
        <circle key={`${x}-${y}`} cx={(x + 0.5) * PITCH} cy={(y + 0.5) * PITCH} r={PITCH * 0.36} fill="currentColor" />
      ))}
    </svg>
  );
}
