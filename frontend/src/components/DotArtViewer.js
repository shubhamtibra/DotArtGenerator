import React from "react";

function DotArtViewer({ dotArt }) {
  return (
    <pre
      className="bg-base02 p-4 rounded border border-base01 overflow-auto animate-fade-in"
      style={{
        fontFamily: "monospace",
        lineHeight: "1em",
        fontSize: "6px", // Adjust font size as needed
        whiteSpace: "pre",
      }}
    >
      {dotArt}
    </pre>
  );
}

export default DotArtViewer;
