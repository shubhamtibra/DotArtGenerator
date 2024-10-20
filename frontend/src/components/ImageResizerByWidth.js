import React, { useState } from "react";
import { Resizable, ResizableBox } from "react-resizable";
import "react-resizable/css/styles.css";

function ImageResizerByWidth({
  imageSrc,
  aspectRatio,
  onWidthChange,
  onAspectRatioChange,
  transformScale,
  resizedImageDimensions,
}) {
  const [sliderWidth, setSliderWidth] = useState(100);

  const handleWidthSliderChange = (e) => {
    setSliderWidth(e.target.value);
    onWidthChange(parseInt(e.target.value));
  };

  return (
    <div className="space-y-4">
      <div className="relative inline-block">
        {/* <ResizableBox
          className="box absolutely-positioned top-aligned left-aligned"
          width={200}
          height={200}
          resizeHandles={["sw", "se", "nw", "ne", "w", "e", "n", "s"]}
        >
          <span className="text">
            {"<ResizableBox> with incorrect scale 1"}
          </span>
        </ResizableBox> */}
        {/* <Resizable
          className="box absolutely-positioned top-aligned left-aligned"
          width={size.width}
          height={size.height}
          onResize={handleResize}
          resizeHandles={["sw", "se", "nw", "ne", "w", "e", "n", "s"]}
          transformScale={0.75}
        >
          <div style={{ width: size.width, height: size.height }}>
            <img
              src={imageSrc}
              alt="Preview"
              style={{ width: "100%", height: "100%", objectFit: "contain" }}
              className="border border-base01"
            />
          </div>
        </Resizable> */}
        <div
          style={{
            width: resizedImageDimensions.width,
            height: resizedImageDimensions.height,
            transform: `scale(${transformScale})`,
          }}
        >
          <img
            src={imageSrc}
            alt="Preview"
            style={{
              width: "100%",
              height: "100%",
              objectFit: "contain",
            }}
            className="border border-base01"
          />
        </div>
      </div>
      <div>
        <label className="block mb-2">Width:</label>
        <input
          type="range"
          min="1"
          max="200"
          value={sliderWidth}
          onChange={handleWidthSliderChange}
          className="w-full cursor-pointer"
        />
        <div className="text-center">
          {" "}
          {`${Math.round(resizedImageDimensions.width, 2)}px x ${Math.round(
            resizedImageDimensions.height,
            2
          )}px`}
        </div>
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

export default ImageResizerByWidth;
