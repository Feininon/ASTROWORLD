import numpy as np
import torch
from keras.models import load_model  # Use Keras for loading .h5 files
import librosa  # For audio processing

# Define emotion labels
emotion_labels = ['happy', 'sad', 'angry', 'neutral', 'calm', 'disgust', 'fearful', 'surprised']

# Load models
def load_rnn_model():
    return load_model(r"C:\Users\hanus\Videos\programming\KCG_main\EMO INTE PRE\nervous_intensity_modelEMO.h5")

def load_fnn_model():
    return torch.load(r"C:\Users\hanus\Videos\programming\KCG_main\KCG_modelFNN\KCG_final_model.pth")

def load_lstm_model():
    return torch.load(r"C:\Users\hanus\Videos\programming\KCG_main\LSTM\KCG_LSTM_model1")

def load_gru_model():
    return torch.load(r"C:\Users\hanus\Videos\programming\KCG_main\KCG_modelGRU\KCG_GRU_model.pth")

# Audio feature extraction
def extract_audio_features(file_path):
    # Load audio file
    audio, sample_rate = librosa.load(file_path, sr=None)
    # Example feature extraction (change this as per your model's requirements)
    features = librosa.feature.mfcc(audio, sr=sample_rate, n_mfcc=13)
    # Flatten and return
    return features.flatten().reshape(1, -1)

def get_fnn_prediction(model, input_data):
    return model(input_data).argmax().item()  # Adjust as needed

def get_gru_prediction(model, input_data):
    return model(input_data).argmax().item()  # Adjust as needed

def get_lstm_prediction(model, input_data):
    return model(input_data).argmax().item()  # Adjust as needed

def get_rnn_prediction(model, input_data):
    return model(input_data).argmax().item()  # Adjust as needed

# Load all models
rnn_model = load_rnn_model()
fnn_model = load_fnn_model()
lstm_model = load_lstm_model()
gru_model = load_gru_model()

# Get audio file path from the user
audio_file = 'sample.wav'

# Extract audio features
input_data = extract_audio_features(audio_file)

# Get predictions from each model
rnn_pred = get_rnn_prediction(rnn_model, input_data)
fnn_pred = get_fnn_prediction(fnn_model, input_data)
lstm_pred = get_lstm_prediction(lstm_model, input_data)
gru_pred = get_gru_prediction(gru_model, input_data)

# Aggregate the predictions
avg_pred = [rnn_pred, fnn_pred, lstm_pred, gru_pred]
predicted_emotion_index = max(set(avg_pred), key=avg_pred.count)

# Map the index to the corresponding emotion label
predicted_emotion = emotion_labels[predicted_emotion_index]

# Output the predicted emotion
print("Predicted Emotion:", predicted_emotion)
