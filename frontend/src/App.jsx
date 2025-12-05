import { useState } from 'react'
import React from 'react'
import { Button } from "@heroui/button"
import './App.css'

function Upload() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);

  const fileInputRef = React.useRef();

  const onFileChange = (event) => {
    const file = event.target.files[0];
    if (!file) return;

    if (!file.type.startsWith("image/")) {
      alert("Please select an image file");
      return;
    }

    setSelectedFile(file);
    setPreview(URL.createObjectURL(file));
  };

	return (
    <div className="upload-wrapper">

      {/* Hidden input */}
      <input
        type="file"
        accept="image/*"
        ref={fileInputRef}
        onChange={onFileChange}
        style={{ display: "none" }}
      />

      <Button
        className="upload-button submit"
        size="md"
        radius="lg"
        onPress={() => fileInputRef.current?.click()}>
        Upload your image!
      </Button>
      
      {/* Preview */}
      {preview && (
        <div className="img-preview">
          <img src={preview} alt="Uploaded preview" />
        </div>
      )}
    </div>	
	);
};

function Text() {
  return (
    <>
      <div className='text'>
        <a>
          Welcome to ZooSnap!
        <br />
          Upload a picture of an animal, and let our AI identify it for you!
        </a>
      </div>
    </>
  );
}

function App() {

  return (
    <>
      <div className='page'>
        <div className='splash'>
          <h1 className='splash-title'> ZooSnap </h1>
        </div>

        <div>
          <Text />
        </div>


        <div className="submit-hover">
          <Upload />
        </div>  
      </div>
    </>
  )
}

export default App
