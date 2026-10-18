/** The app mark: a 3×3 halftone, dots shrinking away from the top-left corner. */
const DOTS = [
  [0, 0, 3.4], [1, 0, 2.8], [2, 0, 2.0],
  [0, 1, 2.8], [1, 1, 2.2], [2, 1, 1.5],
  [0, 2, 2.0], [1, 2, 1.5], [2, 2, 1.0],
];

export function Logo({ className = "h-5 w-5" }) {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" className={className}>
      {DOTS.map(([x, y, r]) => (
        <circle key={`${x}-${y}`} cx={4.5 + x * 7.5} cy={4.5 + y * 7.5} r={r} fill="currentColor" />
      ))}
    </svg>
  );
}
