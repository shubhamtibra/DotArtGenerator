import { act, renderHook } from "@testing-library/react";
import { generateDotArt } from "../api/client";
import { useDotArt } from "./useDotArt";

jest.mock("../api/client", () => ({
  ...jest.requireActual("../api/client"),
  generateDotArt: jest.fn(),
}));

const SOURCE = { id: 1, input: { url: "https://example.com/a.png" } };
const art = (text) => ({ art: text, mode: "braille", columns: 1, rows: 1 });
const flush = () => act(async () => {});

function renderDotArt(options = { width: 10 }, source = SOURCE) {
  return renderHook((props) => useDotArt(props.source, props.options), { initialProps: { source, options } });
}

beforeEach(() => {
  jest.useFakeTimers();
  generateDotArt.mockReset();
});

afterEach(() => {
  jest.useRealTimers();
});

test("debounces rapid changes into one request with the latest options", async () => {
  generateDotArt.mockResolvedValue(art("⣿"));
  const { result, rerender } = renderDotArt();
  rerender({ source: SOURCE, options: { width: 11 } });
  rerender({ source: SOURCE, options: { width: 12 } });
  expect(result.current.isLoading).toBe(true);

  act(() => jest.advanceTimersByTime(300));
  await flush();
  expect(generateDotArt).toHaveBeenCalledTimes(1);
  expect(generateDotArt.mock.calls[0][0]).toBe(SOURCE.input);
  expect(generateDotArt.mock.calls[0][1]).toEqual({ width: 12 });
  expect(result.current.status).toBe("success");
  expect(result.current.result.art).toBe("⣿");
});

test("aborts superseded requests and ignores their late responses", async () => {
  let resolveFirst;
  generateDotArt
    .mockImplementationOnce(() => new Promise((resolve) => (resolveFirst = resolve)))
    .mockResolvedValueOnce(art("new"));
  const { result, rerender } = renderDotArt();
  act(() => jest.advanceTimersByTime(300));

  rerender({ source: SOURCE, options: { width: 20 } });
  expect(generateDotArt.mock.calls[0][2].signal.aborted).toBe(true);
  act(() => jest.advanceTimersByTime(300));
  await flush();
  expect(result.current.result.art).toBe("new");

  await act(async () => resolveFirst(art("stale")));
  expect(result.current.result.art).toBe("new");
});

test("keeps the previous art while loading and never reports aborts as errors", async () => {
  generateDotArt.mockResolvedValueOnce(art("first")).mockImplementation(
    (_source, _options, { signal }) =>
      new Promise((_resolve, reject) => {
        signal.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError")));
      })
  );
  const { result, rerender, unmount } = renderDotArt();
  act(() => jest.advanceTimersByTime(300));
  await flush();

  rerender({ source: SOURCE, options: { width: 30 } });
  act(() => jest.advanceTimersByTime(300));
  rerender({ source: SOURCE, options: { width: 40 } });
  await flush();
  expect(result.current.status).toBe("loading");
  expect(result.current.error).toBeNull();
  expect(result.current.result.art).toBe("first");
  unmount();
});

test("surfaces failures and retries on demand", async () => {
  generateDotArt.mockRejectedValueOnce(Object.assign(new Error("boom"), { code: "server_error" })).mockResolvedValueOnce(art("ok"));
  const { result } = renderDotArt();
  act(() => jest.advanceTimersByTime(300));
  await flush();
  expect(result.current.status).toBe("error");
  expect(result.current.error.message).toBe("boom");

  act(() => result.current.retry());
  act(() => jest.advanceTimersByTime(300));
  await flush();
  expect(result.current.status).toBe("success");
  expect(result.current.error).toBeNull();
});

test("starts fresh for a new source and stays idle when disabled", async () => {
  generateDotArt.mockResolvedValue(art("a"));
  const { result, rerender } = renderHook((props) => useDotArt(props.source, { width: 10 }, { enabled: props.enabled }), {
    initialProps: { source: SOURCE, enabled: true },
  });
  act(() => jest.advanceTimersByTime(300));
  await flush();
  expect(result.current.result.art).toBe("a");

  rerender({ source: { id: 2, input: { url: "https://example.com/b.png" } }, enabled: false });
  expect(result.current.result).toBeNull();
  expect(result.current.status).toBe("idle");
  act(() => jest.advanceTimersByTime(1000));
  expect(generateDotArt).toHaveBeenCalledTimes(1);
});
