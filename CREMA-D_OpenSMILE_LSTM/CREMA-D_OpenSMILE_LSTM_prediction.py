import torch
import torch.nn as nn
import numpy as np
import opensmile

# Define the LSTM model class
class LSTMAudioClassifier(nn.Module):
    def __init__(self, hidden_size, num_layers, num_classes):
        super(LSTMAudioClassifier, self).__init__()
        self.lstm = None  # To be defined dynamically
        self.fc = None    # To be defined dynamically
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

    def initialize_layers(self, input_size):
        self.lstm = nn.LSTM(input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out, _ = self.lstm(x, (h_0, c_0))
        out = out[:, -1, :]  # Take the output of the last time step
        out = self.fc(out)   # Pass through the fully connected layer
        return out

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hidden_size = 128
num_layers = 2
num_classes = 8  # Adjust based on your specific number of classes
model = LSTMAudioClassifier(hidden_size, num_layers, num_classes)

# Load the model state
model_state = torch.load('KCG_LSTM_model1', weights_only=True)
input_size = model_state['lstm.weight_ih_l0'].shape[1]  # Dynamically set input size based on model weights
model.initialize_layers(input_size)
model.load_state_dict(model_state)
model.to(device)
model.eval()

# Function to extract features from an audio file using OpenSMILE
def extract_features(file_path):
    smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02)
    features = smile.process_file(file_path)
    
    # Return the mean of the features
    features_mean = np.mean(features, axis=0)
    return features_mean

# Prediction function
def predict(model, file_paths):
    predictions = []
    
    for file_path in file_paths:
        try:
            features = extract_features(file_path)
            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # Add batch and sequence dimensions
            
            with torch.no_grad():
                outputs = model(features_tensor)
                preds = torch.argmax(outputs, dim=1)
                predictions.append(preds.item())
        
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    return predictions

# Example usage of the prediction function
audio_file_paths = ['sample.wav']  # Replace with your audio file paths
predictions = predict(model, audio_file_paths)

# Assuming you have the label encoder from the training phase
label_encoder = {0: 'Anger', 1: 'Joy', 2: 'Sadness', 3: 'Fear', 4: 'Surprise', 5: 'Neutral',  6: 'Disgust', 7:' Happy'}  # Adjust based on your specific label encoder

predicted_labels = [label_encoder.get(pred) for pred in predictions if pred is not None]
print(predicted_labels)
