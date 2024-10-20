import React, { useState, useEffect } from "react";
import axios from "axios";
import ImageResizer from "./ImageResizer.js";
import DotArtViewer from "./DotArtViewer.js";

function ImageUploader() {
  const [imageFile, setImageFile] = useState(null);
  const [imageUrl, setImageUrl] = useState("");
  const [imageSrc, setImageSrc] = useState(null);
  const [dotArt, setDotArt] = useState("");
  const [width, setWidth] = useState(100);
  const [aspectRatio, setAspectRatio] = useState(null);
  const [debounceTimeout, setDebounceTimeout] = useState(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    setImageFile(file);
    setImageUrl("");
    setDotArt("");
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        setImageSrc(event.target.result);
      };
      reader.readAsDataURL(file);
    } else {
      setImageSrc(null);
    }
  };

  const handleUrlChange = (e) => {
    const url = e.target.value;
    setImageUrl(url);
    setImageFile(null);
    setDotArt("");
    if (url) {
      setImageSrc(url);
    } else {
      setImageSrc(null);
    }
  };

  const debounceGenerateDotArt = () => {
    if (debounceTimeout) {
      clearTimeout(debounceTimeout);
    }
    const timeout = setTimeout(() => {
      generateDotArt();
    }, 1000); // Wait for 1 second after resizing
    setDebounceTimeout(timeout);
  };

  useEffect(() => {
    if (imageSrc) {
      debounceGenerateDotArt();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [width, aspectRatio]);

  const generateDotArt = () => {
    const formData = new FormData();
    if (imageFile) {
      formData.append("file", imageFile);
    } else if (imageUrl) {
      formData.append("url", imageUrl);
    } else {
      return;
    }
    formData.append("width", width);
    if (aspectRatio) {
      formData.append("aspect_ratio", aspectRatio);
    }

    axios
      .post("http://localhost:8000/api/upload/", formData)
      .then((response) => {
        setDotArt(response.data.dot_art);
      })
      .catch((error) => {
        console.error("Error generating dot art:", error);
      });
  };

  return (
    <div className="space-y-8">
      <div className="space-y-4">
        <div>
          <label className="block mb-2 cursor-pointer hover:text-blue transition-colors duration-200">
            Upload Image File:
          </label>
          <input
            type="file"
            onChange={handleFileChange}
            className="block w-full text-base0 cursor-pointer bg-base02 border border-base01 p-2 rounded hover:bg-base01 focus:outline-none focus:ring-2 focus:ring-blue transition"
          />
        </div>
        <div>
          <label className="block mb-2 cursor-pointer hover:text-blue transition-colors duration-200">
            Or Enter Image URL
          </label>
          <input
            type="text"
            value={imageUrl}
            onChange={handleUrlChange}
            className="block w-full bg-base02 border border-base01 p-2 rounded text-base0 placeholder-base1 focus:outline-none focus:ring-2 focus:ring-blue hover:bg-base01 transition"
            placeholder="Enter image URL"
          />
        </div>
      </div>

      {imageSrc && (
        <ImageResizer
          imageSrc={imageSrc}
          width={width}
          aspectRatio={aspectRatio}
          onWidthChange={(e) => setWidth(e.target.value)}
          onAspectRatioChange={(e) => setAspectRatio(e.target.value)}
          onImageChange={debounceGenerateDotArt}
        />
      )}

      {dotArt && <DotArtViewer dotArt={dotArt} />}
    </div>
  );
}

export default ImageUploader;
