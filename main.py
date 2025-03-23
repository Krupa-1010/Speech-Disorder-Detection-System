# 1. Install required libraries
# pip install fastapi uvicorn supabase python-dotenv

# 2. Set up Supabase client in your backend
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from datetime import datetime
from supabase import create_client, Client
import os
from fastapi.middleware.cors import CORSMiddleware

import torch.nn.functional as F

import numpy as np
import requests
from fastapi.responses import JSONResponse

import uvicorn
from processes_auido import ensemble_predict  # Adjust the module name as needed

import librosa
import librosa.display
import matplotlib.pyplot as plt
import io
from PIL import Image
import time
import uuid





# Your Supabase configuration
SUPABASE_URL = "https://zvttbbgrywlnezmoahet.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inp2dHRiYmdyeXdsbmV6bW9haGV0Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0MTAyNDYzNCwiZXhwIjoyMDU2NjAwNjM0fQ.c4xoulWYQZaO7o7QR0mV-Q_PBP1i94hkHxLFNdzLuCM"


# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# 3. Define models for user registration/login
class UserRegistration(BaseModel):
    firstName: str
    lastName: str
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

# 4. Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Add your frontend URL here
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AudioRequest(BaseModel):
    file_url: str
    user_id: str
    audio_id: str

@app.post("/predict/")
async def predict_audio(request: AudioRequest):
    try:
        # Download the audio file
        response = requests.get(request.file_url)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to download audio file")
            
        # Save to temp file
        temp_file = f"temp_{request.audio_id}.wav"
        with open(temp_file, "wb") as f:
            f.write(response.content)
            
        # Run prediction
        result = ensemble_predict(temp_file)  # Your prediction function
        
        # Generate spectrogram
        y, sr = librosa.load(temp_file)
        plt.figure(figsize=(10, 4))
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_dB = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr)
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()
        
        # Save spectrogram to memory
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        
        # Upload spectrogram to Supabase
        spectrogram_path = f"{request.user_id}/{request.audio_id}_spectrogram.png"
        try:
            upload_result = supabase.storage.from_('spectrogram-images').upload(
                spectrogram_path,
                buf.getvalue(),
                file_options={"contentType": "image/png"}
            )
            print(f"✅ Uploaded spectrogram: {spectrogram_path}")
        except Exception as upload_error:
            print(f"❌ Spectrogram upload failed: {str(upload_error)}")
            # Continue without failing the whole request
            spectrogram_path = None
        
        # Store metadata in database
        supabase.table('audio_analyses').insert({
            'user_id': request.user_id,
            'audio_id': request.audio_id,
            'audio_path': request.file_url,
            'spectrogram_path': spectrogram_path,
            'prediction': result['prediction'],
            'confidence': float(result['confidence'])
        }).execute()
        
        # Return results with paths
        return {
            'prediction': result['prediction'],
            'confidence': float(result['confidence']),
            'audio_id': request.audio_id,
            'spectrogram_path': spectrogram_path
        }
        
    except Exception as e:
        import traceback
        print(f"❌ Error details: {str(e)}")
        print(traceback.format_exc())  # Print the full stack trace
        raise HTTPException(status_code=500, detail=str(e))

# Add this route for testing
@app.get("/test-spectrogram/")
async def test_spectrogram():
    try:
        # Generate a simple test waveform
        import numpy as np
        y = np.sin(np.linspace(0, 100, 1000))
        sr = 22050
        
        # Generate spectrogram
        plt.figure(figsize=(10, 4))
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_dB = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr)
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()
        
        # Save spectrogram to memory
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        
        return {"message": "Spectrogram generated successfully"}
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return {"error": str(e)}

# 5. Add routes for user registration and login
@app.post("/auth/register")
async def register_user(user: UserRegistration):
    try:
        # Check if user already exists
        existing_user = supabase.from_('users').select('*').eq('email', user.email).execute()
        
        if len(existing_user.data) > 0:
            return JSONResponse(
                status_code=400,
                content={"error": "User with this email already exists"}
            )
        
        # Insert new user
        new_user = {
            'first_name': user.firstName,
            'last_name': user.lastName,
            'email': user.email,
            'password': user.password,  # In production, hash this password!
        }
        
        # Try to insert the user
        insert_result = supabase.from_('users').insert(new_user).execute()
        
        if insert_result.data:
            # Successfully inserted, now get the user for login
            user_query = supabase.from_('users').select('*').eq('email', user.email).execute()
            if user_query.data and len(user_query.data) > 0:
                user_data = user_query.data[0]
                # Remove password from response
                user_response = {k: v for k, v in user_data.items() if k != 'password'}
                return JSONResponse(
                    status_code=201,
                    content={
                        "message": "User registered successfully",
                        "user": user_response
                    }
                )
        
        return JSONResponse(
            status_code=400,
            content={"error": "Failed to create user"}
        )
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error during registration"}
        )

@app.post("/auth/login")
async def login_user(user: UserLogin):
    try:
        # Find user by email
        result = supabase.from_('users').select('*').eq('email', user.email).execute()
        
        if len(result.data) == 0:
            return JSONResponse(
                status_code=401,
                content={"error": "Invalid email or password"}
            )
            
        # Get user data
        user_data = result.data[0]
        
        # Check password (in production, verify against hashed password)
        if user_data['password'] != user.password:
            return JSONResponse(
                status_code=401,
                content={"error": "Invalid email or password"}
            )
            
        # Return user info (exclude password)
        user_response = {
            "email": user_data['email'],
            "first_name": user_data['first_name'],
            "last_name": user_data['last_name'],
            "id": user_data['id']
        }
        
        return JSONResponse(
            status_code=200,
            content={
                "message": "Login successful",
                "user": user_response
            }
        )
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

# 6. Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    