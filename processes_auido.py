import librosa
import numpy as np
import torch
import torchaudio
import torch.nn.functional as F
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from hybridmodel import HybridModel
from stutteringmodel import StutterDetector



dysarthria_model = HybridModel()
dysarthria_model.load_state_dict(torch.load("saved_model.pth", map_location="cpu"), strict=False)
dysarthria_model.eval()

stuttering_model = StutterDetector()
checkpoint = torch.load("best_stutter_detector.pth", map_location="cpu")
stuttering_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
stuttering_model.eval()

# Initialize the ASR model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
asr_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h") # Use the appropriate model

def preprocess_dysarthria(audio_path, processor, target_spec_shape=(128, 128)):
    waveform, sample_rate = librosa.load(audio_path, sr=16000)
    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", padding="max_length", truncation=True, max_length=160000)
    audio_tensor = inputs.input_values.squeeze(0)
    spectrogram = librosa.feature.melspectrogram(y=waveform, sr=16000)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
    spectrogram_resized = librosa.util.fix_length(spectrogram_db, size=target_spec_shape[1], axis=1)
    if spectrogram_resized.shape[0] != target_spec_shape[0]:
        spectrogram_resized = np.pad(spectrogram_resized, ((0, target_spec_shape[0] - spectrogram_resized.shape[0]), (0, 0)), mode='constant')
    spectrogram_tensor = torch.tensor(spectrogram_resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Ensure (1, 1, H, W) shape
    return audio_tensor, spectrogram_tensor

def preprocess_stuttering(audio_path, processor, max_length=16000*2):
    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if waveform.shape[1] > max_length:
        waveform = waveform[:, :max_length]
    else:
        waveform = torch.nn.functional.pad(waveform, (0, max_length - waveform.shape[1]))
    spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=400, hop_length=160, n_mels=64)(waveform)
    spectrogram = torchaudio.transforms.AmplitudeToDB()(spectrogram)
    spectrogram = (spectrogram - spectrogram.mean()) / spectrogram.std()
    spectrogram = spectrogram.squeeze(0).unsqueeze(0).unsqueeze(0)  # Ensure (1, 1, H, W) format
    inputs = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    return inputs.input_values.squeeze(), spectrogram

# ✅ Phoneme Repetition Detection (ASR-based)
def detect_phoneme_repetition(audio_path):
    print(f"🔹 Running ASR on: {audio_path}")  # Debugging
    waveform, sr = librosa.load(audio_path, sr=16000)
    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = asr_model(inputs.input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    phonemes = transcription.split()

    # Compute repetition score
    repetition_score = sum(1 for i in range(1, len(phonemes)) if phonemes[i] == phonemes[i-1]) / len(phonemes)
    print(f"🔹 Phoneme Repetition Score: {repetition_score}")  # Debugging

    return repetition_score
def extract_speech_features(audio_path):
    waveform, sr = librosa.load(audio_path, sr=16000)

    # Compute phoneme repetition score
    repetition_score = detect_phoneme_repetition(audio_path)

    # Process audio with Wav2Vec2
    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = asr_model(inputs.input_values).logits  # Get model predictions

    # Decode predictions into text
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]  # Convert IDs to words

    words = transcription.split()
    speaking_rate = len(words) / (librosa.get_duration(filename=audio_path) + 1e-5)  # Avoid division by zero

    # Compute pause frequency (silent segments)
    intervals = librosa.effects.split(waveform, top_db=30)  # Detect silent pauses
    pause_count = len(intervals)
    pause_frequency = pause_count / (librosa.get_duration(filename=audio_path) + 1e-5)

    # Handle NaN values
    if np.isnan(repetition_score):
        repetition_score = 0  # Or use dataset mean

    return [repetition_score, speaking_rate, pause_frequency]


# ✅ Prediction Pipeline
def ensemble_predict(audio_path):
    dys_audio, dys_spec = preprocess_dysarthria(audio_path, processor)
    stut_audio, stut_spec = preprocess_stuttering(audio_path, processor)

    dys_audio = dys_audio.unsqueeze(0)
    stut_audio = stut_audio.unsqueeze(0)

    with torch.no_grad():
        dys_output = dysarthria_model(dys_audio, dys_spec)
        stut_output = stuttering_model(stut_audio, stut_spec)

    dys_probs = F.softmax(dys_output, dim=1).squeeze().cpu().numpy()
    stut_probs = F.softmax(stut_output, dim=1).squeeze().cpu().numpy()

    repetition_score, speaking_rate, pause_frequency = extract_speech_features(audio_path)

    # Restored ensemble weighting (previously had 77% accuracy)
    healthy_confidence = (0.2 * dys_probs[0]) + (0.8 * stut_probs[0])
    dys_confidence = (0.5 * dys_probs[1]) + (0.5 * stut_probs[1])
    stut_confidence = (0.3 * dys_probs[1]) + (0.7 * stut_probs[1])

    # Feature-based confidence adjustments
    dys_confidence += 0.2 * (1 - speaking_rate)  # Dysarthria often has slower speech
    stut_confidence += 0.2 * repetition_score  # Stuttering is correlated with repetition

    if repetition_score < 0.02 and speaking_rate < 1.0:
        healthy_confidence += 0.2
    if pause_frequency < 1.2 and speaking_rate < 0.9:
        stut_confidence *= 0.75  # Reduced from 0.8 to 0.75
    if pause_frequency > 2.0 and speaking_rate < 0.7:
        stut_confidence *= 0.8  # Only reduce stuttering confidence in extreme cases
    if repetition_score < 0.01:
        stut_confidence *= 0.9  # Less aggressive reduction compared to 0.75
    if repetition_score < 0.02 and 0.8 < speaking_rate < 1.5:
        healthy_confidence += 0.15  # Increased from 0.1 to 0.15
    if repetition_score < 0.01 and 1.2 < pause_frequency < 2.0:
        healthy_confidence += 0.1  # Added new condition

    if repetition_score > 0.05:
        stut_confidence += 0.3
    if pause_frequency > 1.5:
        stut_confidence += 0.2
    if speaking_rate > 2.0:
        stut_confidence += 0.2
    if speaking_rate < 0.8:
        dys_confidence += 0.2

    if 0.5 < stut_probs[1] < 0.8 and pause_frequency > 0.5:
        stut_confidence += 0.2
    if stut_probs[1] > 0.75:
        dys_confidence *= 0.8
    if abs(stut_confidence - dys_confidence) < 0.05 and healthy_confidence > 0.4:
        healthy_confidence *= 0.8
    
    print(f"Confidence scores -> Healthy: {healthy_confidence:.2f}, Dysarthria: {dys_confidence:.2f}, Stuttering: {stut_confidence:.2f}")
    print(f"Phoneme Repetition Score: {repetition_score:.3f}, Speaking Rate: {speaking_rate:.3f}, Pause Frequency: {pause_frequency:.3f}")

    conf_threshold = 0.02
    sorted_confidences = sorted([(healthy_confidence, 'Healthy'), (dys_confidence, 'Dysarthria'), (stut_confidence, 'Stuttering')], reverse=True)
    top1_conf, top1_label = sorted_confidences[0]
    top2_conf, top2_label = sorted_confidences[1]

    

    if abs(top1_conf - top2_conf) < conf_threshold:
        if top1_label == "Dysarthria" and repetition_score > 0.05:
            return f"Stuttering (Resolved by phoneme repetition score: {repetition_score:.3f})"
        if top1_label == "Stuttering" and repetition_score < 0.05:
            return f"Dysarthria (Resolved by phoneme repetition score: {repetition_score:.3f})"
      # Debugging
    print(f"🔹 Final Prediction: {top1_label} (Confidence: {top1_conf:.2f})")  # Debugging

    return {"prediction": top1_label, "confidence": top1_conf,}