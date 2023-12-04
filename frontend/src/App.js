import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useDropzone } from 'react-dropzone';
import './App.css';

const App = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [result, setResult] = useState(null);
  const [uploadButtonVisible, setUploadButtonVisible] = useState(false);

  const MESSAGE = "However, the results may not be completely accurate, so it's a good idea to ask a professional for a definite answer.";

  const onDrop = (acceptedFiles) => {
    setSelectedFile(acceptedFiles[0]);
    setUploadButtonVisible(true);
  };

  const { getRootProps, getInputProps } = useDropzone({
    onDrop,
    accept: 'image/*',
    multiple: false,
  });

  const handleUploadClick = async () => {
    try {
      if (selectedFile) {
        const formData = new FormData();
        formData.append('image', selectedFile);

        const response = await axios.post('http://localhost:5000/predict_status', formData);

        setResult(response.data.result);
      }
    } catch (error) {
      console.error('Error:', error);
    }
  };

  const handleDeleteClick = () => {
    setSelectedFile(null);
    setUploadButtonVisible(false);
    setResult(null);
  };

  useEffect(() => {
    setUploadButtonVisible(!!selectedFile);
  }, [selectedFile]);

  const handleUploadAndPredictClick = async () => {
    if (!selectedFile) {
      alert('Please select a file first.');
      return;
    }

    await handleUploadClick();
  };

  return (
    <div>
      <nav className="navbar">
        <h1>Brain Tumor Detection</h1>
      </nav>

      <div className="app-container">
        <div className="dropzone-container">
          <div {...getRootProps()} className="dropzone">
            <input {...getInputProps()} />
            {selectedFile ? (
              <>
                <img
                  src={URL.createObjectURL(selectedFile)}
                  alt="Preview"
                  style={{ maxWidth: '100%', maxHeight: '100px', marginBottom: '10px' }}
                />
                <p>{selectedFile.name}</p>
              </>
            ) : (
              <p>Drag & drop an image here, or click to select one</p>
            )}
          </div>
          {selectedFile && (
            <button className="delete-button" onClick={handleDeleteClick}>
              Clear
            </button>
          )}
        </div>

        <button onClick={handleUploadAndPredictClick} className="upload-button">
          Upload and Predict
        </button>

        {result !== null && (
          <div className="result-container" style={{ backgroundColor: result ? 'green' : 'rgb(208, 13, 13)' }}>
            <h2>Prediction Result</h2>
            <p >{result ? `Based on the file you shared, it seems like there might be a tumor. ${MESSAGE}` 
            : `Based on the file you shared, it seems like there might be no tumor. ${MESSAGE}`}</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default App;
