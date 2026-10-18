import { BRAILLE_BLANK, brailleBits, brailleDots, encodeBraille, isBrailleChar } from "./braille";

test("maps each bit to its dot position", () => {
  expect(brailleDots("⠁")).toEqual([[0, 0]]);
  expect(brailleDots("⠈")).toEqual([[1, 0]]);
  expect(brailleDots("⡀")).toEqual([[0, 3]]);
  expect(brailleDots("⢀")).toEqual([[1, 3]]);
  expect(brailleDots("⣿")).toHaveLength(8);
  expect(brailleDots(BRAILLE_BLANK)).toEqual([]);
});

test("encoding and decoding round-trip", () => {
  for (let code = 0x2800; code <= 0x28ff; code += 1) {
    const char = String.fromCharCode(code);
    expect(encodeBraille(brailleDots(char))).toBe(char);
  }
});

test("non-Braille characters have no dots", () => {
  expect(isBrailleChar("@")).toBe(false);
  expect(brailleBits(" ")).toBe(0);
  expect(brailleDots("x")).toEqual([]);
});
