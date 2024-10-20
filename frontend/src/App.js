import React from "react";
import ImageUploader from "./components/ImageUploader";

function App() {
  return (
    <div className="min-h-screen bg-base03 text-base0">
      <div className="container mx-auto p-4">
        <h1 className="text-4xl font-bold mb-8 text-center">
          Dot Art Generato
        </h1>
        <ImageUploader />
      </div>
    </div>
  );
}

export default App;
