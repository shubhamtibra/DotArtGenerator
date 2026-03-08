import React, { useState, useEffect, useRef } from "react";
import axios from "axios";
import DotArtViewer from "./DotArtViewer.js";

function ImageUploader() {
  const [imageFile, setImageFile] = useState(null);
  const [imageSrc, setImageSrc] = useState(null);
  const [debounceTimeout, setDebounceTimeout] = useState(null);
  const [width, setWidth] = useState(100);
  const [imageUrlInput, setImageUrlInput] = useState("");
  const [imageLoaded, setImageLoaded] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const imageObject = useRef(new Image());
  const originalImageSize = useRef(null);
  const [dotArtVariants, setDotArtVariants] = useState([]);
  const [selectedVariantIndex, setSelectedVariantIndex] = useState(0);
  const fileInputRef = useRef(null);

  useEffect(() => {
    imageObject.current.onload = () => {
      originalImageSize.current = {
        width: imageObject.current.naturalWidth,
        height: imageObject.current.naturalHeight,
      };
      setWidth(100); // Default to 100px width
      setImageLoaded(true);
    };
    imageObject.current.onerror = (err) => {
      console.error("Error loading image:", err);
    };
  }, []);

  const handleFile = (file) => {
    if (file && file.type.startsWith('image/')) {
      const localImageUrl = URL.createObjectURL(file);
      imageObject.current.src = localImageUrl;
      setImageSrc(localImageUrl);
      setImageFile(file);
      setImageUrlInput("");
      setDotArtVariants([]);
    }
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) handleFile(file);
  };

  const handleDragEnter = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  };

  const handleUrlChange = (e) => {
    const url = e.target.value;
    setImageUrlInput(url);
  };

  const handleUrlSubmit = () => {
    if (imageUrlInput) {
      imageObject.current.src = imageUrlInput;
      setImageSrc(imageUrlInput);
      setImageFile(null);
      setDotArtVariants([]);
    }
  };

  const debounceGenerateDotArt = () => {
    if (debounceTimeout) {
      clearTimeout(debounceTimeout);
    }
    const timeout = setTimeout(() => {
      generateDotArt();
    }, 800);
    setDebounceTimeout(timeout);
  };

  useEffect(() => {
    if (imageSrc && imageLoaded) {
      debounceGenerateDotArt();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [width, imageSrc, imageLoaded]);

  const handleWidthChange = (e) => {
    const newWidth = parseInt(e.target.value);
    setWidth(newWidth);
  };

  const generateDotArt = () => {
    if (!imageSrc) return;
    
    setIsProcessing(true);
    const formData = new FormData();
    if (imageFile) {
      formData.append("file", imageFile);
    } else {
      formData.append("url", imageSrc);
    }
    formData.append("width", width);
    
    axios
      .post("http://localhost:8000/api/upload/", formData)
      .then((response) => {
        setDotArtVariants(response.data.ascii_art_variants);
        setSelectedVariantIndex(response.data.ascii_art_variants.length - 1); // Set to "Our Pick"
        setIsProcessing(false);
      })
      .catch((error) => {
        console.error("Error generating dot art:", error);
        setIsProcessing(false);
      });
  };

  const getHeight = () => {
    if (!originalImageSize.current) return width;
    return Math.round(
      (width * originalImageSize.current.height) / originalImageSize.current.width
    );
  };

  return (
    <div className="min-h-screen bg-[#0a0a0f] text-white relative overflow-hidden">
      {/* Animated background elements */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-1/4 -left-20 w-96 h-96 bg-purple-600/20 rounded-full blur-[100px] animate-pulse"></div>
        <div className="absolute bottom-1/4 -right-20 w-96 h-96 bg-blue-600/20 rounded-full blur-[100px] animate-pulse" style={{animationDelay: '1s'}}></div>
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-cyan-500/10 rounded-full blur-[120px]"></div>
      </div>

      <div className="relative z-10 max-w-5xl mx-auto px-6 py-12">
        {/* Header */}
        <div className="text-center mb-16 animate-fade-in">
          <div className="inline-block mb-4">
            <span className="px-4 py-2 rounded-full text-xs font-semibold bg-gradient-to-r from-purple-500/20 to-blue-500/20 border border-purple-500/30 text-purple-300 uppercase tracking-wider">
              ✨ AI Powered
            </span>
          </div>
          <h1 className="text-6xl md:text-7xl font-black mb-4 bg-gradient-to-r from-purple-400 via-pink-400 to-cyan-400 bg-clip-text text-transparent animate-gradient">
            Dot Art Generator
          </h1>
          <p className="text-xl text-gray-400 max-w-md mx-auto">
            Transform your images into stunning ASCII masterpieces
          </p>
        </div>

        {/* Upload Section */}
        <div className="glass-card rounded-3xl p-8 mb-10 animate-slide-up" style={{animationDelay: '0.2s'}}>
          {/* Drag and Drop Zone */}
          <div
            onDragEnter={handleDragEnter}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`drop-zone border-2 border-dashed rounded-2xl p-16 text-center cursor-pointer mb-8 ${
              isDragging ? "dragging border-purple-500" : "border-white/10"
            }`}
          >
            <input
              ref={fileInputRef}
              type="file"
              onChange={handleFileChange}
              accept="image/*"
              className="hidden"
            />
            <div className="space-y-6">
              <div className="text-7xl animate-float">🎨</div>
              <div>
                <p className="text-2xl font-bold text-white mb-2">
                  {isDragging ? "Release to upload" : "Drop your image here"}
                </p>
                <p className="text-gray-500">or click to browse your files</p>
              </div>
              <div className="flex items-center justify-center gap-3 text-sm text-gray-500">
                <span className="px-3 py-1 rounded-full bg-white/5">PNG</span>
                <span className="px-3 py-1 rounded-full bg-white/5">JPG</span>
                <span className="px-3 py-1 rounded-full bg-white/5">GIF</span>
                <span className="px-3 py-1 rounded-full bg-white/5">WEBP</span>
              </div>
            </div>
          </div>

          {/* URL Input */}
          <div className="flex gap-4">
            <div className="flex-1 relative">
              <input
                type="text"
                value={imageUrlInput}
                onChange={handleUrlChange}
                onKeyPress={(e) => e.key === 'Enter' && handleUrlSubmit()}
                placeholder="Paste an image URL..."
                className="w-full bg-white/5 border border-white/10 rounded-xl px-5 py-4 text-white placeholder-gray-500 focus:outline-none focus:border-purple-500/50 focus:ring-2 focus:ring-purple-500/20 transition-all"
              />
            </div>
            <button
              onClick={handleUrlSubmit}
              className="btn-primary px-8 py-4 rounded-xl font-bold text-white"
            >
              Load URL
            </button>
          </div>
        </div>

        {/* Image Preview & Controls */}
        {imageLoaded && (
          <div className="grid lg:grid-cols-5 gap-8 mb-10 animate-fade-in">
            {/* Preview - takes 3 columns */}
            <div className="lg:col-span-3 glass-card rounded-3xl p-6">
              <div className="flex items-center gap-3 mb-5">
                <div className="w-3 h-3 rounded-full bg-gradient-to-r from-purple-500 to-blue-500"></div>
                <h3 className="text-lg font-bold text-white">Image Preview</h3>
              </div>
              <div className="bg-black/30 rounded-2xl p-6 flex items-center justify-center" style={{ minHeight: "350px" }}>
                <img
                  src={imageSrc}
                  alt="Preview"
                  className="max-w-full max-h-80 object-contain rounded-xl shadow-2xl"
                />
              </div>
            </div>

            {/* Width Control - takes 2 columns */}
            <div className="lg:col-span-2 glass-card rounded-3xl p-6">
              <div className="flex items-center gap-3 mb-5">
                <div className="w-3 h-3 rounded-full bg-gradient-to-r from-cyan-500 to-blue-500"></div>
                <h3 className="text-lg font-bold text-white">Output Size</h3>
              </div>
              <div className="space-y-8">
                <div>
                  <div className="flex justify-between items-center mb-4">
                    <span className="text-gray-400 text-sm">Width</span>
                    <span className="text-3xl font-black bg-gradient-to-r from-purple-400 to-cyan-400 bg-clip-text text-transparent">
                      {width}px
                    </span>
                  </div>
                  <input
                    type="range"
                    min="20"
                    max="300"
                    value={width}
                    onChange={handleWidthChange}
                    className="w-full mb-4"
                  />
                  <div className="flex justify-between text-xs text-gray-500">
                    <span>20px</span>
                    <span>300px</span>
                  </div>
                </div>

                <div className="p-4 rounded-xl bg-white/5 border border-white/5">
                  <div className="flex justify-between items-center">
                    <span className="text-gray-400">Output Dimensions</span>
                    <span className="font-mono text-cyan-400">{width} × {getHeight()}</span>
                  </div>
                </div>

                {isProcessing && (
                  <div className="flex flex-col items-center justify-center gap-4 py-6">
                    <div className="spinner w-10 h-10"></div>
                    <span className="text-purple-400 font-medium">Generating magic...</span>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Results - Carousel */}
        {dotArtVariants.length > 0 && (
          <div className="glass-card rounded-3xl p-8 animate-slide-up">
            <div className="flex items-center gap-3 mb-6">
              <div className="w-3 h-3 rounded-full bg-gradient-to-r from-green-500 to-cyan-500"></div>
              <h3 className="text-2xl font-bold text-white">Your ASCII Art</h3>
            </div>

            {/* Carousel */}
            <div className="relative px-16">
              {/* Navigation Arrows */}
              <button
                onClick={() => setSelectedVariantIndex(prev => prev > 0 ? prev - 1 : dotArtVariants.length - 1)}
                className="absolute left-0 top-1/2 -translate-y-1/2 z-20 w-14 h-14 rounded-full bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-500 hover:to-blue-500 shadow-lg shadow-purple-500/30 flex items-center justify-center text-white transition-all duration-200 hover:scale-110"
              >
                <svg className="w-7 h-7" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M15 19l-7-7 7-7" />
                </svg>
              </button>
              
              <button
                onClick={() => setSelectedVariantIndex(prev => prev < dotArtVariants.length - 1 ? prev + 1 : 0)}
                className="absolute right-0 top-1/2 -translate-y-1/2 z-20 w-14 h-14 rounded-full bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-500 hover:to-blue-500 shadow-lg shadow-purple-500/30 flex items-center justify-center text-white transition-all duration-200 hover:scale-110"
              >
                <svg className="w-7 h-7" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M9 5l7 7-7 7" />
                </svg>
              </button>

              {/* Variant Label */}
              <div className="text-center mb-4">
                <span className="inline-block px-4 py-2 rounded-full bg-gradient-to-r from-purple-500/20 to-cyan-500/20 border border-purple-500/30 text-white font-medium">
                  {dotArtVariants[selectedVariantIndex].description === "Our Pick" 
                    ? "🌟 Our Pick" 
                    : typeof dotArtVariants[selectedVariantIndex].description === 'number'
                    ? `Gamma: ${dotArtVariants[selectedVariantIndex].description.toFixed(1)}`
                    : dotArtVariants[selectedVariantIndex].description}
                </span>
              </div>

              {/* Art Display */}
              <DotArtViewer 
                dotArt={dotArtVariants[selectedVariantIndex].dot_art} 
                variantName={
                  dotArtVariants[selectedVariantIndex].description === "Our Pick" 
                    ? "our-pick" 
                    : typeof dotArtVariants[selectedVariantIndex].description === 'number'
                    ? `gamma-${dotArtVariants[selectedVariantIndex].description.toFixed(1)}`
                    : dotArtVariants[selectedVariantIndex].description
                }
              />

              {/* Dots Indicator */}
              <div className="flex justify-center gap-2 mt-6">
                {dotArtVariants.map((_, index) => (
                  <button
                    key={index}
                    onClick={() => setSelectedVariantIndex(index)}
                    className={`w-2.5 h-2.5 rounded-full transition-all duration-300 ${
                      index === selectedVariantIndex 
                        ? "bg-gradient-to-r from-purple-500 to-cyan-500 w-8" 
                        : "bg-white/20 hover:bg-white/40"
                    }`}
                  />
                ))}
              </div>

              {/* Variant Counter */}
              <div className="text-center mt-4 text-gray-500 text-sm">
                {selectedVariantIndex + 1} of {dotArtVariants.length} variants
              </div>
            </div>
          </div>
        )}

        {/* Footer */}
        <div className="text-center mt-16 text-gray-600 text-sm">
          <p>Made with 💜 for creative minds</p>
        </div>
      </div>
    </div>
  );
}

export default ImageUploader;
