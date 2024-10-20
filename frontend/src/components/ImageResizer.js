import React, { useState } from "react";
import { Resizable } from "react-resizable";
import "react-resizable/css/styles.css";

function ImageResizer({
  imageSrc,
  width,
  aspectRatio,
  onWidthChange,
  onAspectRatioChange,
  onImageChange,
}) {
  const [size, setSize] = useState({ width: 200, height: 200 });

  const handleResize = (event, { size }) => {
    setSize(size);
    onWidthChange({ target: { value: size.width } });
    if (aspectRatio) {
      const newAspectRatio = size.height / size.width;
      onAspectRatioChange({ target: { value: newAspectRatio } });
    }
    if (onImageChange) {
      onImageChange();
    }
  };

  const handleWidthSliderChange = (e) => {
    const newWidth = parseInt(e.target.value, 10);
    const newHeight = aspectRatio ? newWidth * aspectRatio : size.height;
    setSize({ width: newWidth, height: newHeight });
    onWidthChange(e);
    if (onImageChange) {
      onImageChange();
    }
  };

  return (
    <div className="space-y-4">
      <div className="relative inline-block">
        <Resizable
          width={size.width}
          height={size.height}
          onResize={handleResize}
          minConstraints={[50, 50]}
          maxConstraints={[500, 500]}
          handle={
            <span
              className="react-resizable-handle"
              onMouseUp={onImageChange}
            />
          }
        >
          <div style={{ width: size.width, height: size.height }}>
            <img
              src={imageSrc}
              alt="Preview"
              style={{ width: "100%", height: "100%", objectFit: "contain" }}
              className="border border-base01"
            />
          </div>
        </Resizable>
      </div>
      <div>
        <label className="block mb-2">Width:</label>
        <input
          type="range"
          min="50"
          max="500"
          value={size.width}
          onChange={handleWidthSliderChange}
          className="w-full cursor-pointer"
        />
        <div className="text-center">{Math.round(size.width)}px</div>
      </div>
      <div>
        <label className="block mb-2">Aspect Ratio (Height/Width):</label>
        <input
          type="number"
          value={aspectRatio || ""}
          onChange={onAspectRatioChange}
          step="0.1"
          min="0.1"
          className="w-full bg-base02 border border-base01 p-2 rounded focus:outline-none focus:ring-2 focus:ring-blue"
        />
      </div>
    </div>
  );
}

export default ImageResizer;
