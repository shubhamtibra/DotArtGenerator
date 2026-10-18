import { API_URL } from "../config";

/** An API failure with a user-facing message plus the server's error `code` and HTTP `status`. */
export class ApiError extends Error {
  constructor(message, { code = "unknown_error", status = 0, details = null } = {}) {
    super(message);
    this.name = "ApiError";
    this.code = code;
    this.status = status;
    this.details = details;
  }
}

export function isAbortError(error) {
  return error?.name === "AbortError";
}

const FALLBACK_MESSAGES = {
  400: "The server couldn't process that request.",
  404: "The generator endpoint wasn't found. Check REACT_APP_API_URL.",
  413: "That image is too large for the server (10 MB maximum).",
  429: "Too many requests. Wait a moment, then try again.",
};

function firstDetail(details) {
  if (!details || typeof details !== "object") return null;
  for (const [field, messages] of Object.entries(details)) {
    const message = Array.isArray(messages) ? messages[0] : messages;
    if (typeof message === "string" && message) {
      return field === "non_field_errors" ? message : `${field.replace(/_/g, " ")}: ${message}`;
    }
  }
  return null;
}

function messageFor(status, body) {
  const serverMessage = typeof body?.error === "string" ? body.error.trim() : "";
  if (serverMessage) {
    const detail = body.code === "invalid_options" ? firstDetail(body.details) : null;
    return detail && !serverMessage.includes(detail) ? `${serverMessage} (${detail})` : serverMessage;
  }
  if (FALLBACK_MESSAGES[status]) return FALLBACK_MESSAGES[status];
  if (status >= 500) return "The generator server ran into a problem. Try again in a moment.";
  return `Unexpected response from the server (HTTP ${status}).`;
}

async function readJson(response) {
  try {
    return await response.json();
  } catch (error) {
    if (isAbortError(error)) throw error;
    return null;
  }
}

/** Multipart body for /api/generate/: the image (file or url) plus every option as a string. */
export function buildGenerateBody(source, options) {
  const body = new FormData();
  if (source.file) {
    body.append("file", source.file, source.filename || source.file.name || "image");
  } else {
    body.append("url", source.url);
  }
  for (const [key, value] of Object.entries(options)) {
    if (value !== undefined && value !== null) body.append(key, String(value));
  }
  return body;
}

/**
 * Generate dot art. `source` is `{ file, filename? }` or `{ url }`; `options`
 * uses the API's field names. Rejects with ApiError on failure, or with the
 * fetch AbortError when `signal` aborts (callers should ignore those).
 */
export async function generateDotArt(source, options, { signal } = {}) {
  let response;
  try {
    response = await fetch(`${API_URL}/api/generate/`, {
      method: "POST",
      body: buildGenerateBody(source, options),
      headers: { Accept: "application/json" },
      signal,
    });
  } catch (error) {
    if (isAbortError(error)) throw error;
    throw new ApiError(`Can't reach the generator server at ${API_URL}. Is the backend running?`, {
      code: "network_error",
    });
  }

  const body = await readJson(response);
  if (!response.ok) {
    throw new ApiError(messageFor(response.status, body), {
      code: body?.code || "http_error",
      status: response.status,
      details: body?.details ?? null,
    });
  }
  if (!body || typeof body.art !== "string") {
    throw new ApiError("The server sent a response this app doesn't understand.", {
      code: "bad_response",
      status: response.status,
    });
  }
  return body;
}
