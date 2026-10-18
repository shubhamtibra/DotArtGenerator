import { prepareImageFile, uploadFilename, validateImageFile, validateImageUrl } from "./image";

const MB = 1024 * 1024;

function fileOfSize(name, type, size) {
  const file = new File(["x"], name, { type });
  Object.defineProperty(file, "size", { value: size });
  return file;
}

afterEach(() => {
  delete global.createImageBitmap;
  delete HTMLCanvasElement.prototype.toBlob;
});

test("validates type and size", () => {
  expect(validateImageFile(fileOfSize("a.png", "image/png", 2 * MB))).toBeNull();
  expect(validateImageFile(fileOfSize("scan.tiff", "", MB))).toBeNull();
  expect(validateImageFile(fileOfSize("a.pdf", "application/pdf", MB))).toMatch(/isn't an image/);
  expect(validateImageFile(fileOfSize("huge.jpg", "image/jpeg", 14.2 * MB))).toBe(
    "That image is 14 MB. The limit is 10 MB."
  );
});

test("accepts only http(s) URLs", () => {
  expect(validateImageUrl(" https://example.com/a.png ")).toEqual({ url: "https://example.com/a.png" });
  expect(validateImageUrl("")).toEqual({ error: "Enter an image URL." });
  expect(validateImageUrl("example.com/a.png").error).toMatch(/full URL/);
  expect(validateImageUrl("javascript:alert(1)").error).toMatch(/only http/i);
});

test("names uploads after their encoded type", () => {
  expect(uploadFilename("photo.heic", "image/jpeg")).toBe("photo.jpg");
  expect(uploadFilename("anim.gif", "image/png")).toBe("anim.png");
  expect(uploadFilename("scan.tiff", "image/tiff")).toBe("scan.tiff");
});

test("falls back to the original file when the browser can't decode it", async () => {
  const file = fileOfSize("a.png", "image/png", MB);
  await expect(prepareImageFile(file)).resolves.toEqual({ blob: file, width: null, height: null });
});

test("keeps small images as they are but reports their size", async () => {
  const close = jest.fn();
  global.createImageBitmap = jest.fn(async () => ({ width: 800, height: 600, close }));
  const file = fileOfSize("a.png", "image/png", MB);
  await expect(prepareImageFile(file)).resolves.toEqual({ blob: file, width: 800, height: 600 });
  expect(close).toHaveBeenCalled();
});

test("downscales large images to a 2048 px JPEG on a canvas", async () => {
  global.createImageBitmap = jest.fn(async () => ({ width: 4000, height: 3000, close() {} }));
  const drawImage = jest.fn();
  HTMLCanvasElement.prototype.getContext = () => ({ drawImage });
  HTMLCanvasElement.prototype.toBlob = function toBlob(callback, type, quality) {
    callback(new Blob([`${this.width}x${this.height}`], { type }));
    expect(quality).toBe(0.92);
  };

  const file = fileOfSize("big.jpg", "image/jpeg", 8 * MB);
  const prepared = await prepareImageFile(file);
  expect(prepared.width).toBe(4000);
  expect(prepared.height).toBe(3000);
  expect(prepared.blob.type).toBe("image/jpeg");
  expect(drawImage).toHaveBeenCalledWith(expect.anything(), 0, 0, 2048, 1536);
});

test("keeps transparency-capable sources lossless when downscaling", async () => {
  global.createImageBitmap = jest.fn(async () => ({ width: 3000, height: 3000, close() {} }));
  HTMLCanvasElement.prototype.getContext = () => ({ drawImage() {} });
  HTMLCanvasElement.prototype.toBlob = (callback, type) => callback(new Blob(["x"], { type }));
  const prepared = await prepareImageFile(fileOfSize("logo.webp", "image/webp", MB));
  expect(prepared.blob.type).toBe("image/png");
});
