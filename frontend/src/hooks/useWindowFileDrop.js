import { useEffect, useState } from "react";
import { useLatest } from "./useLatest";

const carriesFiles = (event) => Array.from(event.dataTransfer?.types ?? []).includes("Files");

/** Make the whole window a drop target for files. Returns whether a file is being dragged over it. */
export function useWindowFileDrop(onFile) {
  const [isDragging, setIsDragging] = useState(false);
  const onFileRef = useLatest(onFile);

  useEffect(() => {
    // dragenter/dragleave fire for every child element; count them to know when we truly left.
    let depth = 0;

    const handleEnter = (event) => {
      if (!carriesFiles(event)) return;
      event.preventDefault();
      depth += 1;
      setIsDragging(true);
    };
    const handleOver = (event) => {
      if (!carriesFiles(event)) return;
      event.preventDefault();
      event.dataTransfer.dropEffect = "copy";
    };
    const handleLeave = (event) => {
      if (!carriesFiles(event)) return;
      depth = Math.max(0, depth - 1);
      if (depth === 0) setIsDragging(false);
    };
    const handleDrop = (event) => {
      if (!carriesFiles(event)) return;
      event.preventDefault();
      depth = 0;
      setIsDragging(false);
      const file = event.dataTransfer.files?.[0];
      if (file) onFileRef.current(file);
    };

    window.addEventListener("dragenter", handleEnter);
    window.addEventListener("dragover", handleOver);
    window.addEventListener("dragleave", handleLeave);
    window.addEventListener("drop", handleDrop);
    return () => {
      window.removeEventListener("dragenter", handleEnter);
      window.removeEventListener("dragover", handleOver);
      window.removeEventListener("dragleave", handleLeave);
      window.removeEventListener("drop", handleDrop);
    };
  }, [onFileRef]);

  return isDragging;
}
