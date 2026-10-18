import { copyText } from "./clipboard";

afterEach(() => {
  delete navigator.clipboard;
  delete document.execCommand;
});

test("uses the async Clipboard API when available", async () => {
  const writeText = jest.fn().mockResolvedValue(undefined);
  Object.defineProperty(navigator, "clipboard", { value: { writeText }, configurable: true });
  await expect(copyText("⣿⣿")).resolves.toBe(true);
  expect(writeText).toHaveBeenCalledWith("⣿⣿");
});

test("falls back to a hidden textarea when the Clipboard API is refused", async () => {
  Object.defineProperty(navigator, "clipboard", {
    value: { writeText: jest.fn().mockRejectedValue(new Error("denied")) },
    configurable: true,
  });
  document.execCommand = jest.fn(() => true);
  await expect(copyText("art")).resolves.toBe(true);
  expect(document.execCommand).toHaveBeenCalledWith("copy");
  expect(document.querySelector("textarea")).toBeNull();
});

test("reports failure when nothing can copy", async () => {
  await expect(copyText("art")).resolves.toBe(false);
});
