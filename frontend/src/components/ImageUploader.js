import React, { useState, useEffect, useRef } from "react";
import axios from "axios";
import ImageResizer from "./ImageResizerByWidth.js";
import DotArtViewer from "./DotArtViewer.js";

function ImageUploader() {
  const [isUsingUpload, setIsUsingUpload] = useState(null);
  const [imageFile, setImageFile] = useState(null);
  const [imageSrc, setImageSrc] = useState(null);
  const [dotArt, setDotArt] = useState("");
  const [debounceTimeout, setDebounceTimeout] = useState(null);
  const [resizedImageDimensions, setResizedImageDimensions] = useState(null);
  const [transformScale, setTransformScale] = useState(null);
  const [aspectRatio, setAspectRatio] = useState(null);
  const [imageUrlInput, setImageUrlInput] = useState("");
  const [imageLoaded, setImageLoaded] = useState(false);
  const imageObject = useRef(new Image());
  const originalImageSize = useRef(null);
  const [dotArtVariants, setDotArtVariants] = useState([]);
  const [selectedVariantIndex, setSelectedVariantIndex] = useState(0);
  const [combinedImage, setCombinedImage] = useState(null);
  const [edgeImage, setEdgeImage] = useState(null);
  const [edgeblurred, setBlurredEdgeImage] = useState();
  const [highContrast, setContrastImage] = useState();
  const [grayImage, setGrayImage] = useState();

  useEffect(() => {
    imageObject.current.onload = () => {
      //   setAspectRatio(imageObject.current.naturalHeight / imageObject.current.naturalWidth);
      originalImageSize.current = {
        width: imageObject.current.naturalWidth,
        height: imageObject.current.naturalHeight,
      };
      setResizedImageDimensions({
        width: imageObject.current.naturalWidth,
        height: imageObject.current.naturalHeight,
      });
      const biggerDimension = Math.max(
        imageObject.current.naturalWidth,
        imageObject.current.naturalHeight
      );
      let initialScale;
      if (biggerDimension < 250) {
        initialScale = 2;
      } else if (biggerDimension < 500) {
        initialScale = biggerDimension / 500;
      } else if (biggerDimension < 2000) {
        initialScale = 500 / biggerDimension;
      } else {
        initialScale = 700 / biggerDimension;
      }
      setTransformScale([initialScale, initialScale]);
      imageObject.current.onerror = (err) => {
        console.log(err.stack);
      };
      setImageLoaded(true);
    };
  }, []);
  useEffect(() => {
    setDotArt("");
    setAspectRatio(null);
    if (isUsingUpload === false) {
      setImageFile(null);
    }
  }, [isUsingUpload]);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      const localImageUrl = URL.createObjectURL(file);
      imageObject.current.src = localImageUrl;
      setImageSrc(localImageUrl);
      setIsUsingUpload(true);
      setImageFile(file);
    } else {
      setImageSrc(null);
    }
  };

  const handleUrlChange = (e) => {
    const url = e.target.value;
    if (url) {
      setIsUsingUpload(false);
      setImageSrc(url);
      setImageUrlInput(url);
      imageObject.current.src = url;
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
  }, [resizedImageDimensions, aspectRatio]);

  const handleResize = ({ width, height }) => {
    if (aspectRatio && width) {
      setResizedImageDimensions({
        width: width,
        height: width * aspectRatio,
      });
    } else if (width) {
      setResizedImageDimensions({
        width: width,
        height:
          (width * imageObject.current.naturalHeight) /
          imageObject.current.naturalWidth,
      });
    }
  };
  const handleSliderWidthChange = (value) => {
    handleResize({ width: (imageObject.current.naturalWidth * value) / 100 });
  };
  const handleAspectRatioChange = (value) => {
    setAspectRatio(value);
  };

  const generateDotArt = () => {
    const formData = new FormData();
    if (isUsingUpload) {
      formData.append("file", imageFile);
    } else if (imageSrc) {
      formData.append("url", imageSrc);
    } else {
      return;
    }
    formData.append("width", resizedImageDimensions.width);
    axios
      .post("http://localhost:8000/api/upload/", formData)
      .then((response) => {
        setDotArtVariants(response.data.ascii_art_variants);
        setSelectedVariantIndex(0);
        setEdgeImage(`data:image/png;base64,${response.data.edge_image}`);
        setCombinedImage(
          `data:image/png;base64,${response.data.combined_image}`
        );
        setBlurredEdgeImage(
          `data:image/png;base64,${response.data.edge_blurred_base64}`
        );
        setContrastImage(
          `data:image/png;base64,${response.data.high_contrast_base64}`
        );
        setGrayImage(`data:image/png;base64,${response.data.gray_img_b64}`);
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
            value={imageUrlInput}
            onChange={handleUrlChange}
            className="block w-full bg-base02 border border-base01 p-2 rounded text-base0 placeholder-base1 focus:outline-none focus:ring-2 focus:ring-blue hover:bg-base01 transition"
            placeholder="Enter image URL"
          />
        </div>
      </div>

      {imageLoaded && (
        <ImageResizer
          imageSrc={imageSrc}
          resizedImageDimensions={resizedImageDimensions}
          onWidthChange={handleSliderWidthChange}
          onAspectRatioChange={handleAspectRatioChange}
          onImageChange={debounceGenerateDotArt}
          transformScale={transformScale}
          aspectRatio={aspectRatio}
        />
      )}
      {edgeImage && combinedImage && (
        <div className="flex flex-wrap space-x-4">
          <div>
            <h2 className="text-center mb-2">Edge Detected Image</h2>
            <img
              src={edgeImage}
              alt="Edge Detected"
              className="border border-base01"
              style={{ width: "500px", height: "500px" }}
            />
          </div>
          <div>
            <h2 className="text-center mb-2">Combined Image</h2>
            <img
              src={combinedImage}
              alt="Combined"
              className="border border-base01"
              style={{ width: "500px", height: "500px" }}
            />
          </div>
          <div>
            <h2 className="text-center mb-2">Blurred Edge Image</h2>
            <img
              src={edgeblurred}
              alt="Blurred Edge"
              className="border border-base01"
              style={{ width: "500px", height: "500px" }}
            />
          </div>
          <div>
            <h2 className="text-center mb-2">High Contrast Image</h2>
            <img
              src={highContrast}
              alt="High Contast"
              className="border border-base01"
              style={{ width: "500px", height: "500px" }}
            />
          </div>
          <div>
            <h2 className="text-center mb-2"> Gray Image</h2>
            <img
              src={grayImage}
              alt="High Contast"
              className="border border-base01"
              style={{ width: "500px", height: "500px" }}
            />
          </div>
        </div>
      )}
      {dotArtVariants.length > 0 && (
        <div>
          <label className="block mb-2">Select Gamma Variant:</label>
          <select
            value={selectedVariantIndex}
            onChange={(e) => setSelectedVariantIndex(e.target.value)}
            className="block w-full bg-base02 border border-base01 p-2 rounded text-base0 focus:outline-none focus:ring-2 focus:ring-blue"
          >
            {dotArtVariants.map((variant, index) => (
              <option key={index} value={index}>
                Gamma {variant.gamma}
              </option>
            ))}
          </select>
          <DotArtViewer dotArt={dotArtVariants[selectedVariantIndex].dot_art} />
        </div>
      )}
    </div>
  );
}

export default ImageUploader;
