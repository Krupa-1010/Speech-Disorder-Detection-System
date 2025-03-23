import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import LoginSignup from './components/LoginSignup';
import AudioUpload from './components/AudioUpload';
import LandingPage from './components/LandingPage';
import './App.css';

// Protected route component to check authentication
const ProtectedRoute = ({ children }) => {
  const isAuthenticated = localStorage.getItem('isAuthenticated') === 'true';
  
  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }
  
  return children;
};

function App() {
  return (
    <BrowserRouter>
      <Routes>
        {/* Landing page as the default route */}
        <Route path="/" element={<LandingPage />} />
        
        {/* Login/Signup page */}
        <Route path="/login" element={<LoginSignup />} />
        <Route path="/signup" element={<LoginSignup />} />
        
        {/* Protected Audio Upload route */}
        <Route 
          path="/audio-upload" 
          element={
            <ProtectedRoute>
              <AudioUpload />
            </ProtectedRoute>
          } 
        />
      </Routes>
    </BrowserRouter>
  );
}

export default App;