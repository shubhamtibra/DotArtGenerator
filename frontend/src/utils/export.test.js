import { drawArt } from "./drawArt";
import { artFilename, pngLayout, renderArtPng } from "./export";

function recordingContext() {
  const calls = { arc: 0, rect: 0, fill: 0, fillRect: 0, fillText: [] };
  return {
    calls,
    beginPath() {},
    moveTo() {},
    arc() {
      calls.arc += 1;
    },
    rect() {
      calls.rect += 1;
    },
    fill() {
      calls.fill += 1;
    },
    fillRect() {
      calls.fillRect += 1;
    },
    fillText(text) {
      calls.fillText.push(text);
    },
  };
}

test("builds descriptive, safe file names", () => {
  expect(artFilename("My Photo (1).JPG", { mode: "braille", columns: 100, rows: 42 }, "txt")).toBe(
    "my-photo-1-braille-100x42.txt"
  );
  expect(artFilename("", { mode: "ascii", columns: 80, rows: 30 }, "png")).toBe("dot-art-ascii-80x30.png");
});

test("PNG layout never exceeds 4096 px per side", () => {
  for (const [mode, columns, rows] of [
    ["braille", 400, 400],
    ["ascii", 400, 400],
    ["blocks", 10, 400],
    ["braille", 10, 5],
  ]) {
    const layout = pngLayout({ mode, columns, rows, charAspect: 0.5 });
    expect(layout.width).toBeLessThanOrEqual(4096);
    expect(layout.height).toBeLessThanOrEqual(4096);
    expect(layout.cellHeight / layout.cellWidth).toBeCloseTo(2);
  }
});

test("renderArtPng resolves to null without canvas support", async () => {
  await expect(
    renderArtPng({ art: "⣿", mode: "braille", columns: 1, rows: 1, charAspect: 0.5, theme: { background: "#000", foreground: "#fff" } })
  ).resolves.toBeNull();
});

test("Braille is drawn as one round dot per raised dot, batched into a single fill", () => {
  const context = recordingContext();
  drawArt(context, { art: "⣿⠁\n⠀⢀", mode: "braille", columns: 2, cellWidth: 20, cellHeight: 40, foreground: "#fff" });
  expect(context.calls.arc).toBe(10);
  expect(context.calls.fill).toBe(1);
});

test("ASCII draws every non-space glyph on the grid; blocks draw cells", () => {
  const ascii = recordingContext();
  drawArt(ascii, { art: "@ .\n#-:", mode: "ascii", columns: 3, cellWidth: 8, cellHeight: 16, foreground: "#fff" });
  expect(ascii.calls.fillText).toEqual(["@", ".", "#", "-", ":"]);

  const blocks = recordingContext();
  drawArt(blocks, { art: "█▓▒░ ", mode: "blocks", columns: 5, cellWidth: 8, cellHeight: 16, foreground: "#fff" });
  expect(blocks.calls.fillRect).toBe(4);
});
