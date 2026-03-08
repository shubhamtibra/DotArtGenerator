import React, { useState } from "react";

function DotArtViewer({ dotArt, variantName }) {
  const [copied, setCopied] = useState(false);
  const [isDarkMode, setIsDarkMode] = useState(true);

  const handleCopy = () => {
    navigator.clipboard.writeText(dotArt);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = () => {
    const blob = new Blob([dotArt], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `ascii-art-${variantName || 'output'}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="relative">
      {/* Action buttons */}
      <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
        {/* Dark/Light Mode Toggle */}
        <div className="flex items-center gap-3">
          <span className="text-gray-400 text-sm">Theme:</span>
          <button
            onClick={() => setIsDarkMode(!isDarkMode)}
            className={`relative w-16 h-8 rounded-full transition-all duration-300 ${
              isDarkMode ? 'bg-gray-700' : 'bg-gray-300'
            }`}
          >
            <div
              className={`absolute top-1 w-6 h-6 rounded-full transition-all duration-300 flex items-center justify-center ${
                isDarkMode 
                  ? 'left-9 bg-gray-900 text-yellow-400' 
                  : 'left-1 bg-white text-yellow-500 shadow-md'
              }`}
            >
              {isDarkMode ? (
                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                  <path d="M17.293 13.293A8 8 0 016.707 2.707a8.001 8.001 0 1010.586 10.586z" />
                </svg>
              ) : (
                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 2a1 1 0 011 1v1a1 1 0 11-2 0V3a1 1 0 011-1zm4 8a4 4 0 11-8 0 4 4 0 018 0zm-.464 4.95l.707.707a1 1 0 001.414-1.414l-.707-.707a1 1 0 00-1.414 1.414zm2.12-10.607a1 1 0 010 1.414l-.706.707a1 1 0 11-1.414-1.414l.707-.707a1 1 0 011.414 0zM17 11a1 1 0 100-2h-1a1 1 0 100 2h1zm-7 4a1 1 0 011 1v1a1 1 0 11-2 0v-1a1 1 0 011-1zM5.05 6.464A1 1 0 106.465 5.05l-.708-.707a1 1 0 00-1.414 1.414l.707.707zm1.414 8.486l-.707.707a1 1 0 01-1.414-1.414l.707-.707a1 1 0 011.414 1.414zM4 11a1 1 0 100-2H3a1 1 0 000 2h1z" clipRule="evenodd" />
                </svg>
              )}
            </div>
          </button>
        </div>

        {/* Copy & Download buttons */}
        <div className="flex gap-2">
          <button
            onClick={handleCopy}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 flex items-center gap-2 ${
              isDarkMode 
                ? 'bg-white/10 hover:bg-white/20 text-white/70 hover:text-white' 
                : 'bg-gray-100 hover:bg-gray-200 text-gray-600 hover:text-gray-900'
            }`}
          >
            {copied ? (
              <>
                <svg className="w-4 h-4 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
                Copied!
              </>
            ) : (
              <>
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                </svg>
                Copy
              </>
            )}
          </button>
          <button
            onClick={handleDownload}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 flex items-center gap-2 ${
              isDarkMode
                ? 'bg-purple-500/20 hover:bg-purple-500/30 text-purple-300 hover:text-purple-200'
                : 'bg-purple-100 hover:bg-purple-200 text-purple-600 hover:text-purple-700'
            }`}
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
            </svg>
            Download
          </button>
        </div>
      </div>
      
      {/* Art container */}
      <div className={`rounded-2xl p-6 overflow-auto max-h-[1000px] transition-all duration-300 ${
        isDarkMode ? 'bg-gray-900 border border-gray-700' : 'bg-white border border-gray-200 shadow-lg'
      }`}>
        <pre
          className={`leading-tight select-all transition-colors duration-300 ${
            isDarkMode ? 'text-green-400' : 'text-black'
          }`}
          style={{ 
            fontFamily: "'Courier New', monospace", 
            fontSize: "0.6rem",
            lineHeight: "0.7em",
            letterSpacing: "0.01em",
            ...(isDarkMode && { textShadow: "0 0 8px rgba(74, 222, 128, 0.3)" })
          }}
        >
          {dotArt}
        </pre>
      </div>
    </div>
  );
}

export default DotArtViewer;
