#Speech Disorder Detection Using Deep Learning (Dysarthria & Stuttering)

📌 Project Overview

This project focuses on detecting speech disorders, specifically stuttering and dysarthria, using an ensemble deep learning approach. It integrates Wav2Vec2 for raw audio processing and a CNN for spectrogram analysis, achieving a robust classification system that improves accuracy over independent models.

🚀 Features

Automatic Speech Disorder Detection: Identifies whether a given speech sample is healthy, stuttering, or dysarthric.

Hybrid Model Approach:

Wav2Vec2 extracts deep audio features from raw speech.

CNN processes spectrogram images to capture frequency patterns.

Combined features enhance classification performance.

Feature-Based Adjustments:

Phoneme repetition score for stuttering.

Speaking rate and pause frequency analysis for dysarthria.

Ensemble Model for Improved Accuracy: Combines independent dysarthria and stuttering models with feature-based confidence adjustments.

Full-Stack Implementation:

Frontend: React.js for user interaction.

Backend: FastAPI for API endpoints and processing.

Database: Supabase for storing audio files, spectrograms, and predictions.

📝 Model Architecture

1️⃣ Wav2Vec2-based Feature Extraction

Extracts high-level speech representations from raw audio.

Pretrained on large-scale speech data.

2️⃣ CNN-based Spectrogram Analysis

4 convolutional layers with BatchNorm, ReLU, and MaxPooling.

Processes Mel spectrograms to capture frequency patterns.

3️⃣ Ensemble Classifier

Combines Wav2Vec2 and CNN features via fully connected layers.

Confidence-based decision making using speech metrics (repetition, pauses, rate).
