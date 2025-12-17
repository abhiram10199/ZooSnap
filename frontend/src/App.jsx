import { useState } from 'react'
import React from 'react'
import { Button } from "@heroui/button"
import axios from 'axios'
import './App.css'

  /**
   * Do one thing at a time:
   * 1. Upload a file - Done
   * 2. Show a preview of the file - Done
   * 3. Send the file to the backend API for prediction
   * 4. Show the prediction result
   * 
   */
function StarterText() {
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

function Upload({ setSelectedFile }) {
  // this is supposed to show a preview of the image you uploaded
  const [preview, setPreview] = useState(null);
  const fileInputRef = React.useRef();

  const onFileChange = (event) => {
    const file = event.target.files[0];
    setSelectedFile(file);

   const reader = new FileReader();
    reader.onloadend = () => {
      setPreview(reader.result);
    };
    if (file) {
      reader.readAsDataURL(file);
    }
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


function Prediction({ selectedFile }) {
  const [prediction, setPrediction] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const handlePredict = async () => {
      if (!selectedFile) return;

      setIsLoading(true);
      const formData = new FormData();
      formData.append('file', selectedFile);
      
      await axios.post('http://localhost:8000/predict', formData)
      .then((response) => {
          console.log("API Response:", response.data);
          setPrediction(response.data);
      })
      .catch((error) => {
          setPrediction({ 
              predicted_class: "Error", 
              confidence: 0, 
              message: "Failed to get prediction. Please try again." 
          });
          console.error("API Call Failed:", error);
      })
      .finally(() => {
          setIsLoading(false);
      });

  };

  return (
    <div className="prediction-wrapper">
      {selectedFile && (
        <Button
          className="predict-button"
          size="md"
          radius="lg"
          onPress={handlePredict}
          isDisabled={isLoading}>
          {isLoading ? "Analyzing..." : "Identify Animal"}
        </Button>
      )}

      {prediction && (
        <div className="prediction-result">
          <h2>Prediction: {prediction.predicted_class}</h2>
          <p>Confidence: {(prediction.confidence * 100).toFixed(2)}%</p>
          {prediction.message && <p>{prediction.message}</p>}
        </div>
      )}
    </div>
  );
}

function App() {
  const [selectedFile, setSelectedFile] = useState(null);

  return (
    <>
      <div className='page'>
        <div className='splash'>
          <h1 className='splash-title'> ZooSnap </h1>
        </div>

        <div>
          <StarterText />
        </div>

        <div className="submit-hover">
          <Upload selectedFile={selectedFile} setSelectedFile={setSelectedFile} />
        </div> 
      </div>

      <div>
        <Prediction selectedFile={selectedFile} />
      </div>
    </>
  )
}

export default App
