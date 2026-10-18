import { useCallback, useEffect, useRef, useState } from "react";
import { generateDotArt, isAbortError } from "../api/client";
import { GENERATE_DEBOUNCE_MS } from "../constants";

const INITIAL_STATE = { status: "idle", result: null, error: null };

/**
 * Live dot-art generation for a source and a set of API options.
 *
 * Every change to the source or options schedules a debounced request; the
 * in-flight request is aborted as soon as anything changes, and responses
 * from superseded requests are ignored. The previous result is kept while a
 * new one loads so the UI can show it dimmed.
 *
 * status: "idle" | "loading" | "success" | "error"
 */
export function useDotArt(source, options, { enabled = true, delay = GENERATE_DEBOUNCE_MS } = {}) {
  const [state, setState] = useState(INITIAL_STATE);
  const [attempt, setAttempt] = useState(0);
  const latestRequest = useRef(0);

  const sourceId = source?.id ?? null;
  const input = source?.input ?? null;
  const optionsKey = JSON.stringify(options);

  // A new image starts from a clean slate rather than showing the old art.
  useEffect(() => {
    setState(INITIAL_STATE);
  }, [sourceId]);

  useEffect(() => {
    const requestId = ++latestRequest.current;
    if (!input || !enabled) {
      setState((current) => (current.status === "loading" ? { ...current, status: "idle" } : current));
      return undefined;
    }

    const controller = new AbortController();
    setState((current) => ({ ...current, status: "loading", error: null }));
    const timer = setTimeout(async () => {
      try {
        const result = await generateDotArt(input, JSON.parse(optionsKey), { signal: controller.signal });
        if (requestId === latestRequest.current) setState({ status: "success", result, error: null });
      } catch (error) {
        if (isAbortError(error) || requestId !== latestRequest.current) return;
        setState((current) => ({ ...current, status: "error", error }));
      }
    }, delay);

    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [input, optionsKey, enabled, delay, attempt]);

  const retry = useCallback(() => setAttempt((count) => count + 1), []);
  const dismissError = useCallback(
    () => setState((current) => ({ ...current, error: null, status: current.result ? "success" : "idle" })),
    []
  );

  return { ...state, isLoading: state.status === "loading", retry, dismissError };
}
