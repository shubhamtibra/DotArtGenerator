import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import App from "./App";
import { ApiError, generateDotArt } from "./api/client";

jest.mock("./api/client", () => ({
  ...jest.requireActual("./api/client"),
  generateDotArt: jest.fn(),
}));

const RESULT = {
  art: "⣿⣷⡄\n⠀⠁⠃",
  mode: "braille",
  columns: 3,
  rows: 2,
  options: { mode: "braille", width: 100, threshold: 131, threshold_auto: true, char_aspect: 0.5 },
  source: { width: 1024, height: 768, format: "PNG" },
  color_map: null,
  processed_image: null,
  elapsed_ms: 37,
};

const DEFAULT_OPTIONS = {
  mode: "braille",
  width: 100,
  dither: "floyd_steinberg",
  invert: true, // the preview background defaults to dark
  brightness: 0,
  contrast: 0,
  gamma: 1,
  sharpen: 0,
  edges: 0,
  equalize: false,
  normalize: true,
  color: false,
  include_processed: false,
};

const findArt = (art) => screen.findByText((_, element) => element.tagName === "PRE" && element.textContent === art);
const lastOptions = () => generateDotArt.mock.calls[generateDotArt.mock.calls.length - 1][1];

async function loadSample(name = /spheres/i) {
  userEvent.click(screen.getByRole("button", { name }));
  await waitFor(() => expect(generateDotArt).toHaveBeenCalled());
}

beforeEach(() => {
  generateDotArt.mockReset();
  global.fetch = jest.fn(async () => ({
    ok: true,
    status: 200,
    blob: async () => new Blob(["png-bytes"], { type: "image/png" }),
  }));
});

afterEach(() => {
  delete global.fetch;
  window.history.replaceState(null, "", "/");
});

describe("empty state", () => {
  test("offers every way to bring in an image", () => {
    render(<App />);
    expect(screen.getByRole("heading", { name: /turn any image into dot art/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /browse/i })).toBeInTheDocument();
    expect(screen.getByRole("textbox", { name: /image url/i })).toBeInTheDocument();
    for (const sample of [/spheres/i, /mountains/i, /waves/i]) {
      expect(screen.getByRole("button", { name: sample })).toBeInTheDocument();
    }
    expect(generateDotArt).not.toHaveBeenCalled();
  });

  test("rejects non-http URLs inline without calling the API", () => {
    render(<App />);
    userEvent.type(screen.getByRole("textbox", { name: /image url/i }), "ftp://example.com/cat.png{enter}");
    expect(screen.getByText(/only http:\/\/ and https:\/\/ urls/i)).toBeInTheDocument();
    expect(generateDotArt).not.toHaveBeenCalled();
  });

  test("rejects files that aren't images", async () => {
    render(<App />);
    userEvent.upload(screen.getByLabelText("Upload image"), new File(["hello"], "notes.txt", { type: "text/plain" }));
    expect(await screen.findByRole("alert")).toHaveTextContent(/isn't an image/i);
    expect(generateDotArt).not.toHaveBeenCalled();
  });
});

describe("generation", () => {
  test("a sample is uploaded once and rendered with the default options", async () => {
    generateDotArt.mockResolvedValue(RESULT);
    render(<App />);
    await loadSample();

    expect(global.fetch).toHaveBeenCalledWith("/samples/spheres.png");
    const [source, options] = generateDotArt.mock.calls[0];
    expect(source.file).toBeInstanceOf(Blob);
    expect(source.filename).toBe("spheres.png");
    expect(options).toEqual(DEFAULT_OPTIONS);

    expect(await findArt(RESULT.art)).toBeInTheDocument();
    expect(screen.getByText("3 × 2")).toBeInTheDocument();
    expect(screen.getByText("37 ms")).toBeInTheDocument();
  });

  test("a chosen file is sent with the current settings", async () => {
    generateDotArt.mockResolvedValue(RESULT);
    render(<App />);
    const file = new File(["image-bytes"], "cat.jpg", { type: "image/jpeg" });
    userEvent.upload(screen.getByLabelText("Upload image"), file);

    await waitFor(() => expect(generateDotArt).toHaveBeenCalledTimes(1));
    const [source, options] = generateDotArt.mock.calls[0];
    expect(source.file).toBe(file); // no canvas in jsdom: the original is uploaded as-is
    expect(options).toEqual(DEFAULT_OPTIONS);
    expect(await screen.findByText("cat.jpg")).toBeInTheDocument();
  });

  test("switching mode regenerates with mode-specific options and the same upload", async () => {
    generateDotArt.mockResolvedValue(RESULT);
    render(<App />);
    await loadSample();
    await findArt(RESULT.art);

    userEvent.click(screen.getByRole("radio", { name: "ASCII" }));
    await waitFor(() => expect(generateDotArt).toHaveBeenCalledTimes(2));
    expect(lastOptions()).toEqual({ ...DEFAULT_OPTIONS, mode: "ascii", charset: "standard", char_aspect: 0.5 });
    expect(generateDotArt.mock.calls[1][0]).toBe(generateDotArt.mock.calls[0][0]);

    userEvent.click(screen.getByRole("radio", { name: "Blocks" }));
    await waitFor(() => expect(generateDotArt).toHaveBeenCalledTimes(3));
    expect(lastOptions()).toEqual({ ...DEFAULT_OPTIONS, mode: "blocks", char_aspect: 0.5 });
  });

  test("the light preview background flips the invert sent to the API", async () => {
    generateDotArt.mockResolvedValue(RESULT);
    render(<App />);
    await loadSample();

    userEvent.click(screen.getByRole("radio", { name: "Light background" }));
    await waitFor(() => expect(generateDotArt).toHaveBeenCalledTimes(2));
    expect(lastOptions().invert).toBe(false);

    userEvent.click(screen.getByRole("switch", { name: "Invert" }));
    await waitFor(() => expect(generateDotArt).toHaveBeenCalledTimes(3));
    expect(lastOptions().invert).toBe(true);
  });

  test("presets apply their settings, including the threshold for undithered Braille", async () => {
    generateDotArt.mockResolvedValue(RESULT);
    render(<App />);
    await loadSample();

    userEvent.click(screen.getByRole("button", { name: /crisp lines/i }));
    await waitFor(() => expect(generateDotArt).toHaveBeenCalledTimes(2));
    expect(lastOptions()).toMatchObject({ mode: "braille", dither: "none", threshold: "auto", edges: 0.6, sharpen: 1 });
    expect(screen.getByRole("button", { name: /crisp lines/i })).toHaveAttribute("aria-pressed", "true");
  });

  test("errors show a banner whose Retry regenerates", async () => {
    generateDotArt
      .mockRejectedValueOnce(
        new ApiError("Can't reach the generator server at http://localhost:8000. Is the backend running?", {
          code: "network_error",
        })
      )
      .mockResolvedValueOnce(RESULT);
    render(<App />);
    await loadSample();

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent(/can't reach the generator server/i);
    expect(alert).toHaveTextContent(/manage\.py runserver/i);

    userEvent.click(within(alert).getByRole("button", { name: /retry/i }));
    await waitFor(() => expect(generateDotArt).toHaveBeenCalledTimes(2));
    expect(await findArt(RESULT.art)).toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  test("an error can be dismissed", async () => {
    generateDotArt.mockRejectedValue(new ApiError("Image could not be decoded.", { code: "invalid_image", status: 400 }));
    render(<App />);
    await loadSample();

    const alert = await screen.findByRole("alert");
    userEvent.click(within(alert).getByRole("button", { name: /dismiss/i }));
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  test("?sample=<name> loads that sample on start", async () => {
    generateDotArt.mockResolvedValue(RESULT);
    window.history.replaceState(null, "", "/?sample=waves");
    render(<App />);
    await waitFor(() => expect(generateDotArt).toHaveBeenCalledTimes(1));
    expect(global.fetch).toHaveBeenCalledWith("/samples/waves.png");
    expect(await screen.findByText("waves.png")).toBeInTheDocument();
  });

  test("removing the image returns to the empty state", async () => {
    generateDotArt.mockResolvedValue(RESULT);
    render(<App />);
    await loadSample();
    await findArt(RESULT.art);

    userEvent.click(screen.getByRole("button", { name: "Remove image" }));
    expect(screen.getByRole("heading", { name: /turn any image into dot art/i })).toBeInTheDocument();
    expect(URL.revokeObjectURL).toHaveBeenCalledWith("blob:mock-preview");
  });
});

describe("export", () => {
  test("Copy puts the art on the clipboard and confirms", async () => {
    const writeText = jest.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", { value: { writeText }, configurable: true });
    generateDotArt.mockResolvedValue(RESULT);
    render(<App />);
    await loadSample();
    await findArt(RESULT.art);

    userEvent.click(screen.getByRole("button", { name: "Copy" }));
    await waitFor(() => expect(writeText).toHaveBeenCalledWith(RESULT.art));
    expect(await screen.findAllByText("Copied to clipboard")).not.toHaveLength(0);
    delete navigator.clipboard;
  });

  test("Download TXT saves the art as UTF-8 text", async () => {
    const click = jest.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});
    generateDotArt.mockResolvedValue(RESULT);
    render(<App />);
    await loadSample();
    await findArt(RESULT.art);

    userEvent.click(screen.getByRole("button", { name: "Download TXT" }));
    const blob = URL.createObjectURL.mock.calls[URL.createObjectURL.mock.calls.length - 1][0];
    expect(blob.type).toBe("text/plain;charset=utf-8");
    expect(click).toHaveBeenCalledTimes(1);
    expect((await screen.findAllByText("Saved spheres-braille-3x2.txt")).length).toBeGreaterThan(0);
    click.mockRestore();
  });

  test("Download PNG explains when the browser can't draw", async () => {
    generateDotArt.mockResolvedValue(RESULT);
    render(<App />);
    await loadSample();
    await findArt(RESULT.art);

    userEvent.click(screen.getByRole("button", { name: "Download PNG" }));
    expect((await screen.findAllByText(/png export isn't supported/i)).length).toBeGreaterThan(0);
  });
});
