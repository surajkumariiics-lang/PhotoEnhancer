import { useState, useRef } from 'react';
import axios from 'axios';
import './App.css';

const API_URL = 'http://localhost:8000';

function App() {
  const [originalImage, setOriginalImage] = useState(null);
  const [enhancedImage, setEnhancedImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [strength, setStrength] = useState(0.7);
  const [format, setFormat] = useState('AUTO');
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileInput = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = (file) => {
    if (!file.type.startsWith('image/')) {
      setError('Please upload an image file');
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      setOriginalImage({
        file: file,
        preview: e.target.result
      });
      setEnhancedImage(null);
      setError(null);
    };
    reader.readAsDataURL(file);
  };

  const handleEnhance = async () => {
    if (!originalImage) return;

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', originalImage.file);
      formData.append('strength', strength.toString());
      formData.append('format', format);

      const response = await axios.post(`${API_URL}/enhance`, formData, {
        responseType: 'blob',
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      const imageUrl = URL.createObjectURL(response.data);
      setEnhancedImage(imageUrl);
    } catch (err) {
      console.error('Enhancement error:', err);
      setError(err.response?.data?.detail || 'Enhancement failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = () => {
    if (!enhancedImage) return;

    const link = document.createElement('a');
    link.href = enhancedImage;
    link.download = `enhanced_${originalImage.file.name}`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleReset = () => {
    setOriginalImage(null);
    setEnhancedImage(null);
    setError(null);
    setStrength(0.7);
    setFormat('AUTO');
  };

  return (
    <div className="app">
      <div className="container">
        <header className="header">
          <div className="logo">
            <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
              <rect width="32" height="32" rx="8" fill="url(#gradient)" />
              <path d="M16 8L20 12H18V20H14V12H12L16 8Z" fill="white" />
              <defs>
                <linearGradient id="gradient" x1="0" y1="0" x2="32" y2="32">
                  <stop offset="0%" stopColor="#667eea" />
                  <stop offset="100%" stopColor="#764ba2" />
                </linearGradient>
              </defs>
            </svg>
            <h1>AI Image Enhancer</h1>
          </div>
          <p className="subtitle">Ultra-HD enhancement • Format selection • ~100KB files</p>
        </header>

        {!originalImage ? (
          <div
            className={`upload-zone ${dragActive ? 'drag-active' : ''}`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleFileInput}
              style={{ display: 'none' }}
            />
            <svg width="64" height="64" viewBox="0 0 64 64" fill="none">
              <rect x="8" y="16" width="48" height="40" rx="4" stroke="currentColor" strokeWidth="2" />
              <circle cx="20" cy="28" r="4" fill="currentColor" />
              <path d="M8 48L20 36L28 44L40 32L56 48" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
            </svg>
            <h2>Drop your image here</h2>
            <p>or click to browse</p>
            <span className="file-types">Supports JPEG, PNG, and more</span>
          </div>
        ) : (
          <div className="workspace">
            <div className="controls">
              <div className="strength-control">
                <label>
                  Enhancement Strength: <strong>{(strength * 100).toFixed(0)}%</strong>
                </label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={strength}
                  onChange={(e) => setStrength(parseFloat(e.target.value))}
                  disabled={loading}
                />
                <div className="strength-labels">
                  <span>Subtle</span>
                  <span>Moderate</span>
                  <span>Strong</span>
                </div>
              </div>

              <div className="format-control">
                <label>
                  Output Format: <strong>{format}</strong>
                </label>
                <select
                  value={format}
                  onChange={(e) => setFormat(e.target.value)}
                  disabled={loading}
                  className="format-select"
                >
                  <option value="AUTO">AUTO (Best compression)</option>
                  <option value="WEBP">WebP (Modern, smallest)</option>
                  <option value="JPEG">JPEG (Universal)</option>
                  <option value="PNG">PNG (Lossless)</option>
                </select>
                <div className="format-info">
                  {format === 'AUTO' && <span>🤖 Automatically selects best format</span>}
                  {format === 'WEBP' && <span>🚀 Modern format, excellent compression</span>}
                  {format === 'JPEG' && <span>📱 Universal compatibility</span>}
                  {format === 'PNG' && <span>🎯 Lossless quality, larger files</span>}
                </div>
              </div>

              <div className="action-buttons">
                <button
                  className="btn btn-primary"
                  onClick={handleEnhance}
                  disabled={loading}
                >
                  {loading ? (
                    <>
                      <span className="spinner"></span>
                      Enhancing...
                    </>
                  ) : (
                    <>
                      <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                        <path d="M8 2L10 6H6L8 2Z" fill="currentColor" />
                        <path d="M8 14L6 10H10L8 14Z" fill="currentColor" />
                        <circle cx="8" cy="8" r="2" fill="currentColor" />
                      </svg>
                      Enhance Image
                    </>
                  )}
                </button>

                {enhancedImage && (
                  <button className="btn btn-secondary" onClick={handleDownload}>
                    <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                      <path d="M8 2V10M8 10L5 7M8 10L11 7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
                      <path d="M2 12V13C2 13.5523 2.44772 14 3 14H13C13.5523 14 14 13.5523 14 13V12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
                    </svg>
                    Download
                  </button>
                )}

                <button className="btn btn-ghost" onClick={handleReset}>
                  Reset
                </button>
              </div>
            </div>

            {error && (
              <div className="error-message">
                <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
                  <circle cx="10" cy="10" r="8" stroke="currentColor" strokeWidth="2" />
                  <path d="M10 6V10M10 13V14" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
                </svg>
                {error}
              </div>
            )}

            <div className="comparison">
              <div className="image-panel">
                <h3>Original</h3>
                <div className="image-container">
                  <img src={originalImage.preview} alt="Original" />
                </div>
              </div>

              {enhancedImage && (
                <div className="image-panel">
                  <h3>Enhanced</h3>
                  <div className="image-container">
                    <img src={enhancedImage} alt="Enhanced" />
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        <footer className="footer">
          <p>Powered by Real-ESRGAN • Ultra-HD quality • Ultra compression (~100KB)</p>
        </footer>
      </div>
    </div>
  );
}

export default App;
