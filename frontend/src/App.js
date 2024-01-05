import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useDropzone } from 'react-dropzone';
import './App.css';

const App = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [type, setType] = useState(null);
  const [uploadButtonVisible, setUploadButtonVisible] = useState(false);
  const [result, setResult] = useState(null);
  const MESSAGE = "Ancak sonuçlar tam olarak doğru olmayabilir, bu yüzden kesin bir cevap almak için doktora görünmek iyi bir fikir olabilir";

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

        const response = await axios.post('http://localhost:5000/predict_category', formData);

        setType(response.data.type);
        if (response.data.type.includes('Yok')) {
          setResult(true);
        } else {
          setResult(false);
        }
      }
    } catch (error) {
      console.error('Error:', error);
    }
  };

  const handleDeleteClick = () => {
    setSelectedFile(null);
    setUploadButtonVisible(false);
    setType(null);
    setResult(null);
  };

  useEffect(() => {
    setUploadButtonVisible(!!selectedFile);
  }, [selectedFile]);

  const handleUploadAndPredictClick = async () => {
    if (!selectedFile) {
      alert('Lütfen bir resim seçin');
      return;
    }

    await handleUploadClick();
  };

  return (
    <div>
      <nav className="navbar">
        <h1>Beyin Tümör Analizi</h1>
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
                  style={{ maxWidth: '100%', maxHeight: '10rem', marginBottom: '10px' }}
                />
                <p>{selectedFile.name}</p>
              </>
            ) : (
              <h4>Bir resmi buraya sürükleyip bırakın veya seçmek için tıklayın</h4>
            )}
          </div>
          {selectedFile && (
            <button className="delete-button" onClick={handleDeleteClick}>
              Temizle
            </button>
          )}
        </div>

        <button onClick={handleUploadAndPredictClick} className="upload-button">
          Yükle ve Tahmin Et
        </button>

        {result !== null && (
          <div className="result-container" style={{ backgroundColor: result ? '#27ae60' : '#c0392b' }}>
            <h2>Tahmin Sonucu</h2>
            <h3 >{type}</h3>
            <h4 >{!result ? `Paylaştığınız dosyaya dayanarak, bir tümör olabileceğini söylemek mümkün görünüyor. ${MESSAGE}` 
            : `Paylaştığınız dosyaya dayanarak herhangi bir tümör bulgusuna rastlanmadı. ${MESSAGE}`}</h4>
          </div>
        )}
      </div>
    </div>
  );
};

export default App;
