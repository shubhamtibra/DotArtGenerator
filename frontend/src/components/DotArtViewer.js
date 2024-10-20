import React from "react";

function DotArtViewer({ dotArt }) {
  return (
    <pre
      className="bg-base02 p-4 rounded border border-base01 overflow-auto animate-fade-in"
      style={{ fontFamily: "monospace", lineHeight: "0.7em" }}
    >
      {dotArt}
    </pre>
  );
}

export default DotArtViewer;
