import { useEffect } from "react";
import { isImageFile, validateImageUrl } from "../utils/image";
import { useLatest } from "./useLatest";

function isEditable(target) {
  return Boolean(target?.closest?.("input, textarea, select, [contenteditable='true']"));
}

function imageFromClipboard(data) {
  const fromFiles = Array.from(data.files ?? []).find(isImageFile);
  if (fromFiles) return fromFiles;
  const item = Array.from(data.items ?? []).find((entry) => entry.kind === "file" && entry.type.startsWith("image/"));
  return item?.getAsFile() ?? null;
}

/**
 * Paste anywhere: an image on the clipboard is loaded as a file; a bare
 * http(s) URL is loaded as a URL unless the user is typing in a field.
 */
export function useClipboardPaste({ onFile, onUrl }) {
  const handlers = useLatest({ onFile, onUrl });

  useEffect(() => {
    const handlePaste = (event) => {
      const data = event.clipboardData;
      if (!data) return;
      const file = imageFromClipboard(data);
      if (file) {
        event.preventDefault();
        handlers.current.onFile(file);
        return;
      }
      if (isEditable(event.target)) return;
      const text = data.getData("text/plain").trim();
      if (!/^https?:\/\/\S+$/i.test(text)) return;
      const { url } = validateImageUrl(text);
      if (url) {
        event.preventDefault();
        handlers.current.onUrl(url);
      }
    };
    window.addEventListener("paste", handlePaste);
    return () => window.removeEventListener("paste", handlePaste);
  }, [handlers]);
}
