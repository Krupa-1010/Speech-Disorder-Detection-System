import React from 'react';
import { useNavigate } from 'react-router-dom';
import speakImage from '../images/speak.webp';
import './LandingPage.css';

const LandingPage = () => {
  const navigate = useNavigate();

  return (
    <div className="landing-container">
      <div className="hero-section" style={{ backgroundImage: `url(${speakImage})` }}>
        <div className="hero-content">
          <h1 className="main-title">
            <span className="highlight">Speech Disorder</span>
            <br />
            Detection System
          </h1>
          <p className="hero-subtitle">Advanced AI-Powered Speech Analysis</p>
        </div>
      </div>

      <div className="landing-content">
        <div className="feature-section">
          <h2>🗣️ Speak Freely, Understand Clearly! 🔍</h2>
          <p>Our AI-powered speech analysis tool detects Dysarthria and Stuttering in real-time, helping with early diagnosis and better communication. Let AI assist you in finding clarity in every word!</p>
        </div>
        
        <div className="disorders-section">
          <h2>Supported Disorders</h2>
          <div className="disorder-cards">
            <div className="disorder-card">
              <div className="card-icon">🗣️🔄</div>
              <h3>Stuttering</h3>
              <p>"Does your speech get stuck on certain sounds? Our AI detects stuttering patterns, offering insights for smoother communication."</p>
            </div>
            <div className="disorder-card">
              <div className="card-icon">🗣️💬</div>
              <h3>Dysarthria</h3>
              <p>"Struggling with slurred or slow speech? Our AI identifies Dysarthria early, helping in speech therapy and rehabilitation."</p>
            </div>
            <div className="disorder-card">
              <div className="card-icon">✅</div>
              <h3>Healthy Speech</h3>
              <p>"Clear and confident speech! Our AI confirms when no speech disorder is detected, reassuring fluent communication."</p>
            </div>
          </div>
        </div>

        <div className="cta-section">
          <h2>🔐 Your Voice, Your Identity!</h2>
          <p className="cta-subtitle">Log in or sign up to detect speech disorders early and get personalized AI-driven insights!"</p>
          <div className="button-group">
            <button className="login-btn" onClick={() => navigate('/login')}>
              Login
            </button>
            <button className="signup-btn" onClick={() => navigate('/signup')}>
              Sign Up
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LandingPage; 