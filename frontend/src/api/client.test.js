import { API_URL } from "../config";
import { ApiError, generateDotArt, isAbortError } from "./client";

const RESULT = { art: "⣿", mode: "braille", columns: 1, rows: 1 };

function respond(status, body) {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: body === undefined ? async () => Promise.reject(new SyntaxError("Unexpected token <")) : async () => body,
  };
}

beforeEach(() => {
  global.fetch = jest.fn();
});

afterEach(() => {
  delete global.fetch;
});

test("posts the file and every option as multipart fields", async () => {
  global.fetch.mockResolvedValue(respond(200, RESULT));
  const file = new Blob(["bytes"], { type: "image/png" });

  await expect(
    generateDotArt({ file, filename: "cat.png" }, { mode: "braille", width: 120, invert: true, threshold: "auto", custom: undefined })
  ).resolves.toEqual(RESULT);

  const [url, init] = global.fetch.mock.calls[0];
  expect(url).toBe(`${API_URL}/api/generate/`);
  expect(init.method).toBe("POST");
  const body = init.body;
  expect(body.get("file").name).toBe("cat.png");
  expect(body.get("mode")).toBe("braille");
  expect(body.get("width")).toBe("120");
  expect(body.get("invert")).toBe("true");
  expect(body.get("threshold")).toBe("auto");
  expect(body.has("custom")).toBe(false);
  expect(body.has("url")).toBe(false);
});

test("sends a URL source as the url field", async () => {
  global.fetch.mockResolvedValue(respond(200, RESULT));
  await generateDotArt({ url: "https://example.com/cat.png" }, { mode: "ascii" });
  const body = global.fetch.mock.calls[0][1].body;
  expect(body.get("url")).toBe("https://example.com/cat.png");
  expect(body.has("file")).toBe(false);
});

test("normalizes server errors, including the first validation detail", async () => {
  global.fetch.mockResolvedValue(
    respond(400, {
      error: "Invalid options.",
      code: "invalid_options",
      details: { width: ["Ensure this value is less than or equal to 400."] },
    })
  );
  const error = await generateDotArt({ url: "https://example.com/a.png" }, {}).catch((e) => e);
  expect(error).toBeInstanceOf(ApiError);
  expect(error.code).toBe("invalid_options");
  expect(error.status).toBe(400);
  expect(error.message).toBe("Invalid options. (width: Ensure this value is less than or equal to 400.)");
});

test("falls back to a friendly message when the error body isn't JSON", async () => {
  global.fetch.mockResolvedValue(respond(502));
  const error = await generateDotArt({ url: "https://example.com/a.png" }, {}).catch((e) => e);
  expect(error.status).toBe(502);
  expect(error.code).toBe("http_error");
  expect(error.message).toMatch(/ran into a problem/i);
});

test("explains network failures in terms of the configured server", async () => {
  global.fetch.mockRejectedValue(new TypeError("Failed to fetch"));
  const error = await generateDotArt({ url: "https://example.com/a.png" }, {}).catch((e) => e);
  expect(error.code).toBe("network_error");
  expect(error.message).toBe(`Can't reach the generator server at ${API_URL}. Is the backend running?`);
});

test("lets aborts through untouched so callers can ignore them", async () => {
  const abort = new DOMException("The operation was aborted.", "AbortError");
  global.fetch.mockRejectedValue(abort);
  const error = await generateDotArt({ url: "https://example.com/a.png" }, {}).catch((e) => e);
  expect(error).toBe(abort);
  expect(isAbortError(error)).toBe(true);
});

test("rejects successful responses without art", async () => {
  global.fetch.mockResolvedValue(respond(200, { hello: "world" }));
  await expect(generateDotArt({ url: "https://example.com/a.png" }, {})).rejects.toMatchObject({ code: "bad_response" });
});
