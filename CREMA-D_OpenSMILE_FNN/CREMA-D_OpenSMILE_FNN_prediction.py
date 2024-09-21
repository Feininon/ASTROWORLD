import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Define the FeedForward Neural Network
class FeedForwardNN(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super(FeedForwardNN, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 128)  # First fully connected layer
        self.fc2 = torch.nn.Linear(128, 64)  # Second fully connected layer
        self.fc3 = torch.nn.Linear(64, num_classes)  # Output layer
        self.relu = torch.nn.ReLU()  # Activation function
        self.dropout = torch.nn.Dropout(0.5)  # Dropout layer for regularization

    def forward(self, x):
        x = self.relu(self.fc1(x))  # Pass through fc1 and apply ReLU
        x = self.dropout(self.relu(self.fc2(x)))  # Pass through fc2, apply ReLU and Dropout
        x = self.fc3(x)  # Output layer (logits)
        return x

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = 988  # Adjust based on your number of features
num_classes = 6  # Adjust based on your specific number of classes
model = FeedForwardNN(input_size, num_classes)
model.load_state_dict(torch.load('KCG_final_model.pth'))
model.to(device)
model.eval()

# Recreate the scaler and label encoder using the original training data
data = pd.read_csv('yio.csv')  # Load your original data
X = data.drop(columns=['emotion_label']).values  # Features
y = data['emotion_label'].values  # Labels

# Create LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # Fit the label encoder

# Create StandardScaler
scaler = StandardScaler()
scaler.fit(X)  # Fit the scaler on the original features

# Function to extract features from an audio file
def extract_features(file_path):
    # Implement your feature extraction logic here
    features = np.random.rand(input_size)  # Placeholder: Replace with actual feature extraction
    return features

# Function for making predictions
def predict(model, scaler, file_path):
    features = extract_features(file_path)
    features = features.reshape(1, -1)  # Reshape to (1, input_size)
    
    # Normalize the features
    features_scaled = scaler.transform(features)

    # Convert to tensor and move to device
    features_tensor = torch.tensor(features_scaled, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        outputs = model(features_tensor)
        preds = torch.argmax(outputs, dim=1)
    
    return preds.cpu().numpy()

# Example usage
audio_file_path = 'sample.wav'  # Replace with your audio file path
predictions = predict(model, scaler, audio_file_path)

# Convert predictions to original labels
predicted_labels = label_encoder.inverse_transform(predictions)
print(predicted_labels)
