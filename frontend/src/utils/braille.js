/** Unicode Braille patterns: U+2800 + an 8-bit mask, one bit per dot in a 2×4 cell. */

export const BRAILLE_BASE = 0x2800;
export const BRAILLE_BLANK = String.fromCharCode(BRAILLE_BASE);

/** [bit, column, row] for each of the eight dots. */
export const BRAILLE_DOTS = [
  [0x01, 0, 0],
  [0x02, 0, 1],
  [0x04, 0, 2],
  [0x08, 1, 0],
  [0x10, 1, 1],
  [0x20, 1, 2],
  [0x40, 0, 3],
  [0x80, 1, 3],
];

export function isBrailleChar(char) {
  const code = char.charCodeAt(0);
  return code >= BRAILLE_BASE && code <= BRAILLE_BASE + 0xff;
}

/** Bit mask of the raised dots, or 0 for anything that isn't a Braille pattern. */
export function brailleBits(char) {
  return isBrailleChar(char) ? char.charCodeAt(0) - BRAILLE_BASE : 0;
}

/** Positions ([column, row]) of the raised dots in a Braille character. */
export function brailleDots(char) {
  const bits = brailleBits(char);
  return BRAILLE_DOTS.filter(([bit]) => bits & bit).map(([, x, y]) => [x, y]);
}

/** Build a Braille character from [column, row] dot positions. */
export function encodeBraille(dots) {
  const bits = dots.reduce((mask, [x, y]) => {
    const dot = BRAILLE_DOTS.find(([, dx, dy]) => dx === x && dy === y);
    return dot ? mask | dot[0] : mask;
  }, 0);
  return String.fromCharCode(BRAILLE_BASE + bits);
}
