import { useCallback, useEffect, useRef, useState } from "react";
import { PUBLIC_URL } from "../config";
import { prepareImageFile, probeImageSize, uploadFilename, validateImageFile } from "../utils/image";

export function sampleImageUrl(sample) {
  return `${PUBLIC_URL}/samples/${sample.id}.png`;
}

export function sampleThumbnailUrl(sample) {
  return `${PUBLIC_URL}/samples/${sample.id}-thumb.jpg`;
}

/**
 * The current source image and its lifecycle.
 *
 * A source is `{ id, kind, name, detail, type, size, width, height, previewUrl, input }`
 * where `input` is what the API client uploads: `{ file, filename }` (prepared
 * once, reused for every regeneration) or `{ url }`. Object URLs are revoked
 * when the source is replaced, removed or the component unmounts. Loads that
 * are superseded by a newer one are discarded.
 */
export function useImageSource() {
  const [source, setSource] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const loadToken = useRef(0);
  const nextId = useRef(1);
  const objectUrl = useRef(null);

  const setObjectUrl = useCallback((url) => {
    if (objectUrl.current) URL.revokeObjectURL(objectUrl.current);
    objectUrl.current = url;
  }, []);

  useEffect(() => () => setObjectUrl(null), [setObjectUrl]);

  /** Begin a load, superseding any in flight. Returns the token that identifies it. */
  const startLoad = useCallback(() => {
    loadToken.current += 1;
    setError(null);
    setIsLoading(true);
    return loadToken.current;
  }, []);

  const loadFile = useCallback(
    async (file, { kind = "file", name } = {}) => {
      const problem = validateImageFile(file);
      if (problem) {
        setError(problem);
        return;
      }
      const token = startLoad();
      const prepared = await prepareImageFile(file);
      if (token !== loadToken.current) return;

      const previewUrl = URL.createObjectURL(prepared.blob);
      setObjectUrl(previewUrl);
      setSource({
        id: nextId.current++,
        kind,
        name: name || file.name || "Untitled image",
        detail: null,
        type: file.type,
        size: file.size,
        width: prepared.width,
        height: prepared.height,
        previewUrl,
        input: { file: prepared.blob, filename: uploadFilename(file.name, prepared.blob.type) },
      });
      setIsLoading(false);
    },
    [setObjectUrl, startLoad]
  );

  const loadSample = useCallback(
    async (sample) => {
      const token = startLoad();
      let file;
      try {
        const response = await fetch(sampleImageUrl(sample));
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const blob = await response.blob();
        file = new File([blob], `${sample.id}.png`, { type: blob.type || "image/png" });
      } catch {
        if (token === loadToken.current) {
          setError(`Couldn't load the “${sample.label}” sample. Check your connection and try again.`);
          setIsLoading(false);
        }
        return;
      }
      if (token !== loadToken.current) return;
      await loadFile(file, { kind: "sample" });
    },
    [loadFile, startLoad]
  );

  /** Use a remote image; `url` must already be validated (see validateImageUrl). */
  const loadUrl = useCallback(
    (url) => {
      const token = startLoad();
      const id = nextId.current++;
      setObjectUrl(null);
      setSource({
        id,
        kind: "url",
        name: new URL(url).hostname,
        detail: url,
        type: null,
        size: null,
        width: null,
        height: null,
        previewUrl: url,
        input: { url },
      });
      setIsLoading(false);
      probeImageSize(url).then((size) => {
        if (size && token === loadToken.current) {
          setSource((current) => (current?.id === id ? { ...current, ...size } : current));
        }
      });
    },
    [setObjectUrl, startLoad]
  );

  const clear = useCallback(() => {
    loadToken.current += 1;
    setObjectUrl(null);
    setSource(null);
    setIsLoading(false);
    setError(null);
  }, [setObjectUrl]);

  const dismissError = useCallback(() => setError(null), []);

  return { source, isLoading, error, loadFile, loadSample, loadUrl, clear, dismissError };
}
