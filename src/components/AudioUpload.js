import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import './AudioUpload.css';
// Import Supabase client
import { createClient } from '@supabase/supabase-js';
// Import jsPDF and html2canvas
import jsPDF from 'jspdf';

const supabaseUrl = 'https://zvttbbgrywlnezmoahet.supabase.co';
const supabaseKey = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inp2dHRiYmdyeXdsbmV6bW9haGV0Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0MTAyNDYzNCwiZXhwIjoyMDU2NjAwNjM0fQ.c4xoulWYQZaO7o7QR0mV-Q_PBP1i94hkHxLFNdzLuCM'; // Replace with your public anon key
const supabase = createClient(supabaseUrl, supabaseKey);

const AudioUpload = () => {
  const navigate = useNavigate();
  const [selectedFile, setSelectedFile] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState('');
  const [prediction, setPrediction] = useState(null);

  // Helper function to convert confidence to level
  const getConfidenceLevel = (confidence) => {
    if (confidence >= 0.7) return 'High';
    if (confidence >= 0.4) return 'Moderate';
    return 'Low';
  };
  
  // Get user data from localStorage
  const userData = JSON.parse(localStorage.getItem('userData') || '{}');
  const userId = userData.id; // Get user ID
  const userEmail = userData.email || 'User';
  const firstName = userData.first_name || 'User';
  const lastName = userData.last_name || 'User';

  console.log('User Data:', {
    userId,
    userEmail,
    firstName,
    lastName,
    rawUserData: userData
  });
  
  useEffect(() => {
    // Check if user is authenticated
    const isAuthenticated = localStorage.getItem('isAuthenticated') === 'true';
    if (!isAuthenticated) {
      navigate('/');
    }
  }, [navigate]);

  const handleLogout = () => {
    // Clear authentication data
    localStorage.removeItem('isAuthenticated');
    localStorage.removeItem('userData');
    
    // Redirect to login page
    navigate('/');
  };

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file && file.type.includes('audio')) {
      setSelectedFile(file);
    }
  };

  const uploadAudio = async () => {
    if (!selectedFile) {
      alert("❌ Please select an audio file.");
      return;
    }
    
    if (!userId) {
      alert("❌ User not authenticated properly. Please log in again.");
      navigate('/');
      return;
    }

    setResult("🔄 Uploading file, please wait...");
    setIsAnalyzing(true);

    try {
      console.log("🔹 Uploading file:", selectedFile.name);

      // Generate a unique audio ID
      const audioId = crypto.randomUUID();
      
      const filePath = `${userId}/${audioId}_${selectedFile.name}`;
      
      // Upload audio file to Supabase Storage
      const { data, error } = await supabase.storage
        .from("auidofiles")
        .upload(filePath, selectedFile, { upsert: true });

      if (error) throw error;

      console.log("✅ File uploaded:", data);

      // Get Public URL
      const { data: urlData, error: urlError } = supabase
        .storage
        .from("auidofiles")
        .getPublicUrl(filePath);

      if (urlError) throw urlError;

      const publicUrl = urlData.publicUrl;
      console.log("🔗 Public URL:", publicUrl);

      // Send file URL to FastAPI for prediction
      console.log("🔹 Sending API request to FastAPI...");
      setResult("🔄 Processing audio, please wait...");

      let response = await fetch("http://127.0.0.1:8000/predict/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          file_url: publicUrl, 
          user_id: userId,
          audio_id: audioId
        })
      });

      console.log("✅ Response Status:", response.status);

      if (!response.ok) {
        throw new Error(`Server error: ${response.status} - ${response.statusText}`);
      }

      let predictionResult = await response.json();
      setPrediction(predictionResult);
      console.log("✅ Prediction Result:", predictionResult);

      // Modify the uploadAudio function where we create the result HTML
      let resultHTML = `
        <div class="prediction-card">
          <div class="prediction-label">Prediction</div>
          <h3 class="prediction-value">${predictionResult.prediction}</h3>
          
          <div class="prediction-label">Confidence Level</div>
          <h3 class="prediction-value">${getConfidenceLevel(predictionResult.confidence)}</h3>
          <div class="confidence-meter">
            <div class="confidence-level" style="width: ${predictionResult.confidence * 100}%"></div>
          </div>
        </div>
      `;
      
      // Add spectrogram if available
      if (predictionResult.spectrogram_path) {
        console.log("Spectrogram available at path:", predictionResult.spectrogram_path);
        
        // Get public URL for the spectrogram
        const { data: spectrogramData } = supabase.storage
          .from("spectrogram-images")
          .getPublicUrl(predictionResult.spectrogram_path);
            
        if (spectrogramData && spectrogramData.publicUrl) {
          resultHTML += `
            <div class="spectrogram-container">
              <h4>Speech Pattern Visualization</h4>
              <img src="${spectrogramData.publicUrl}" alt="Spectrogram" class="spectrogram-image" />
            </div>
          `;
        }
      }
      
      // Update the result element
      setResult(resultHTML);

    } catch (error) {
      console.error("❌ Fetch Error:", error);
      setResult("❌ Error: " + error.message);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const downloadPDF = async (prediction) => {
    const pdf = new jsPDF();
    const date = new Date().toLocaleDateString();
    
    // Create a title and user information section
    pdf.setFontSize(20);
    pdf.text("Audio Analysis Report", 20, 20);
    pdf.setFontSize(12);
    pdf.text(`User: ${userEmail}`, 20, 30);
    pdf.text(`Date of Analysis: ${date}`, 20, 40);
    
    // Add analysis results
    pdf.setFontSize(16);
    pdf.text("Analysis Results", 20, 60);
    pdf.setFontSize(12);
    pdf.text(`Prediction: ${prediction.prediction}`, 20, 70);
    pdf.text(`Confidence Level: ${getConfidenceLevel(prediction.confidence)}`, 20, 80);
    
    // Add spectrogram if available
    if (prediction.spectrogram_path) {
      const { data: spectrogramData } = supabase.storage
        .from("spectrogram-images")
        .getPublicUrl(prediction.spectrogram_path);
        
      if (spectrogramData && spectrogramData.publicUrl) {
        const img = new Image();
        img.crossOrigin = "Anonymous"; // Set crossOrigin to avoid CORS issues
        img.src = spectrogramData.publicUrl;

        img.onload = async () => {
          console.log("Spectrogram image loaded successfully.");
          
          // Create a canvas element
          const canvas = document.createElement('canvas');
          const ctx = canvas.getContext('2d');
          canvas.width = img.width;
          canvas.height = img.height;
          ctx.drawImage(img, 0, 0);

          // Convert canvas to image data
          const imgData = canvas.toDataURL('image/png');
          pdf.addImage(imgData, 'PNG', 20, 90, 170, 90);
          pdf.save(`Speech_Analysis_${firstName}_${lastName}_${prediction.prediction}_${new Date().toISOString().slice(0, 10)}_${new Date().toTimeString().slice(0, 8).replace(/:/g, '-')}.pdf`);
        };

        img.onerror = () => {
          console.error("❌ Error loading spectrogram image.");
          pdf.save(`Speech_Analysis_${firstName}_${lastName}_${prediction.prediction}_${new Date().toISOString().slice(0, 10)}_${new Date().toTimeString().slice(0, 8).replace(/:/g, '-')}.pdf`);
        };
      } else {
        console.error("❌ Spectrogram data is not available.");
        pdf.save(`Speech_Analysis_${firstName}_${lastName}_${prediction.prediction}_${new Date().toISOString().slice(0, 10)}_${new Date().toTimeString().slice(0, 8).replace(/:/g, '-')}.pdf`);
      }

    } else {
      pdf.save(`Speech_Analysis_${firstName}_${lastName}_${prediction.prediction}_${new Date().toISOString().slice(0, 10)}_${new Date().toTimeString().slice(0, 8).replace(/:/g, '-')}.pdf`);
    }
  };

  // Modify the download button to only call downloadPDF when clicked
  const handleDownloadPDF = () => {
    if (prediction) {
      downloadPDF(prediction);
    } else {
      console.error("❌ No prediction data available for PDF download.");
    }
  };

  return (
    <div className="audio-upload-main">
      <div className="upload-container">
        {/* User Info and Logout Button */}
        <div className="user-info">
          <p>Logged in as: <strong>{userEmail}</strong></p>
          <button className="logout-btn" onClick={handleLogout}>Logout</button>
        </div>
        
        <div className="header">
          <div className="waveform-icon">
            <span></span>
            <span></span>
            <span></span>
            <span></span>
            <span></span>
          </div>
          <h2 className="header-title">Audio Analysis</h2>
        </div>
        
        <p>Upload an audio recording to detect speech disorders</p>
        
        {/* Custom File Input */}
        <label className="file-input-label">
          <div className="file-icon">
            <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
              <polyline points="17 8 12 3 7 8"></polyline>
              <line x1="12" y1="3" x2="12" y2="15"></line>
            </svg>
          </div>
          <span className="file-input-text">
            {selectedFile ? 'Change file' : 'Click to upload audio file or drag & drop'}
          </span>
          <input 
            type="file" 
            id="audioFile"
            accept="audio/*" 
            onChange={handleFileChange}
          />
          {selectedFile && (
            <div className="file-name">{selectedFile.name}</div>
          )}
        </label>
        
        {/* Analyze Button */}
        <button 
          onClick={uploadAudio} 
          disabled={isAnalyzing || !selectedFile}
          className={`analyze-btn ${isAnalyzing ? 'analyzing' : ''}`}
        >
          {isAnalyzing ? (
            <>
              <span className="loading-spinner"></span>
              Analyzing...
            </>
          ) : 'Analyze Audio'}
        </button>
        
        {/* Add Download PDF Button */}
        <button 
          onClick={handleDownloadPDF} 
          className="download-pdf-btn"
          disabled={!result || !prediction}
        >
          Download PDF Report
        </button>
        
        <div id="result" dangerouslySetInnerHTML={{ __html: result }}></div>
      </div>
    </div>
  );
};

export default AudioUpload;