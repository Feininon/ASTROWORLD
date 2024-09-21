import os
import pandas as pd
import torch
import numpy as np
import subprocess
import torchaudio
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch.nn as nn

# Load the trained model and scaler
model_save_path = 'KCG_GRU_model.pth'
scaler = StandardScaler()  # Ensure you use the same scaler used during training
label_encoder = LabelEncoder()  # Load your trained label encoder

# Load your trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GRUNetwork(nn.Module):
    def __init__(self, input_size, num_classes):
        super(GRUNetwork, self).__init__()
        self.gru = nn.GRU(input_size, 128, batch_first=True)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add sequence dimension
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out

input_size = 15  # Adjust based on your feature extraction
num_classes = 6  # Adjust to your number of emotion classes
model = GRUNetwork(input_size, num_classes)
model.load_state_dict(torch.load(model_save_path))
model.to(device)

# Function to extract features using OpenSMILE
def extract_features_with_opensmile(audio_file):
    output_file = "features.csv"

    # Run OpenSMILE command using a default config
    command = f"SMILExtract -C {os.path.join(os.environ['OPENSMILE_ROOT'], 'config', 'emo_9s.conf')} -I {audio_file} -O {output_file}"
    subprocess.run(command, shell=True)

    # Load the extracted features
    features = pd.read_csv(output_file)
    return features.values

# Function to predict emotion
def predict_emotion(audio_file_path):
    features = extract_features_with_opensmile(audio_file_path)
    
    # Scale features
    features_scaled = scaler.transform(features)  # Adjust based on your feature shape

    # Convert to tensor
    features_tensor = torch.tensor(features_scaled, dtype=torch.float32).to(device)

    # Add batch dimension and make prediction
    model.eval()
    with torch.no_grad():
        features_tensor = features_tensor.unsqueeze(0)  # Add batch dimension
        output = model(features_tensor)
        predicted_index = torch.argmax(output, dim=1).item()
    
    # Decode the predicted label
    predicted_label = label_encoder.inverse_transform([predicted_index])[0]
    
    return predicted_label

# Example usage
nervousness = predict_emotion(r"C:\Users\hanus\Videos\programming\KCG_main\KCG_model1\resampling\03-01-01-01-01-02-24.wav")
print(f'The predicted emotion is: {nervousness}')
